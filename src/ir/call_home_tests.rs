use super::*;
use crate::ir::expr::{BinOp, BlockRef, Reg, Width};
use crate::ir::lift::{BoxedMem, Region};

fn id(handler: u64) -> BlockId { BlockId { handler, vip: None } }
fn block(id: BlockId, events: Vec<Event>, terminator: Terminator) -> Block {
    Block { id, block_ref: BlockRef(id.handler as u32), params: Vec::new(), events,
        from_vm_context:false, exit_regs:Vec::new(), terminator, cost:0,preds:Vec::new() }
}
fn cfg(arena: Arena, blocks: Vec<Block>) -> Cfg {
    Cfg {entry:blocks[0].id,arena,blocks,total_steps:0,unresolved:0,back_edges:Vec::new(),
        trivial_phis_removed:0,timed_out:false,stack_base:0x700000000,native_entry_proofs: Default::default(), vm_range:(0,0)}
}
fn origin(block: BlockId, index: usize) -> CallOrigin {
    CallOrigin {block,event_index:index,site:0x9000+index as u64}
}
fn result(arena: &mut Arena, origin: CallOrigin) -> Ref {
    let value=arena.opaque("call_ret",Width::W64);
    arena.bind_call_origin(value,origin);value
}
fn call(origin: CallOrigin, value: Ref) -> Event {
    Event::Call {target:None,site:origin.site,rsp:None,import_slot:None,args:Vec::new(),ret:Some(value)}
}
fn linear() -> (Cfg,CallOrigin,Ref) {
    let mut arena=Arena::new();let origin=origin(id(1),0);let value=result(&mut arena,origin);
    let graph=cfg(arena,vec![block(id(1),vec![call(origin,value)],Terminator::Jump(id(2))),
        block(id(2),Vec::new(),Terminator::Return {dest:value})]);
    (graph,origin,value)
}

#[test]
fn call_home_linear_old_return_survives_another_call() {
    let (mut graph,old,value)=linear();
    let later=origin(id(1),1);let later_value=result(&mut graph.arena,later);
    graph.blocks[0].events.push(call(later,later_value));
    graph.blocks[0].exit_regs.push((Reg::Rax,later_value));
    graph.blocks[1].exit_regs.push((Reg::Rbx,value));
    let plan=CallHomePlan::build(&graph);
    assert_eq!(plan.slot_for(old),Some(0));assert_eq!(plan.slot_for(later),None);
    assert_eq!(plan.len(),1);
}

#[test]
fn call_home_checks_same_block_uses_even_with_valid_crossblock_use() {
    for in_target in [false,true] {
        let (mut graph,old,value)=linear();
        let Event::Call {target,args,..}=&mut graph.blocks[0].events[0] else {unreachable!()};
        if in_target {*target=Some(value);} else {args.push((Reg::Rcx,value));}
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),None);
    }
}

#[test]
fn call_home_read_in_earlier_event_rejects_entire_origin() {
    let mut arena=Arena::new();let old=origin(id(1),1);let value=result(&mut arena,old);
    let first=Event::Call {target:None,site:0x8888,rsp:None,import_slot:None,
        args:vec![(Reg::Rcx,value)],ret:None};
    let graph=cfg(arena,vec![block(id(1),vec![first,call(old,value)],Terminator::Jump(id(2))),
        block(id(2),Vec::new(),Terminator::Return {dest:value})]);
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),None);
}

#[test]
fn call_home_missing_duplicate_and_mismatched_definitions_reject() {
    for change in 0..4 {
        let (mut graph,old,value)=linear();
        match change {
            0=>graph.blocks[0].events.clear(),
            1=>graph.blocks[0].events.push(call(old,value)),
            2=>if let Event::Call {site,..}=&mut graph.blocks[0].events[0] {*site+=1;},
            _=>{graph.blocks[0].events.clear();graph.blocks[1].events.push(call(old,value));},
        }
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),None,"change {change}");
    }
}

#[test]
fn call_home_rejects_non_dominating_sibling_definition() {
    let (mut graph,old,value)=linear();
    graph.blocks.insert(0,block(id(0),Vec::new(),Terminator::Branch {
        predicate:value,taken:id(1),not_taken:id(2)}));graph.entry=id(0);
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),None);
}

#[test]
fn call_home_accepts_join_when_unique_definition_dominates() {
    let (mut graph,old,value)=linear();
    let condition=graph.arena.init_reg(Reg::Rcx);
    graph.blocks[0].terminator=Terminator::Branch {predicate:condition,taken:id(2),not_taken:id(3)};
    graph.blocks[1].terminator=Terminator::Jump(id(4));
    graph.blocks.push(block(id(3),Vec::new(),Terminator::Jump(id(4))));
    graph.blocks.push(block(id(4),Vec::new(),Terminator::Return {dest:value}));
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
}

#[test]
fn call_home_rejects_cyclic_definition_and_loop_carried_use() {
    for definition_cycle in [false,true] {
        let (mut graph,old,value)=linear();
        graph.blocks[1].terminator=Terminator::Branch {predicate:value,taken:id(3),
            not_taken:if definition_cycle {id(1)} else {id(2)}};
        graph.blocks.push(block(id(3),Vec::new(),Terminator::Return {dest:value}));
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),if definition_cycle {None} else {Some(0)});
    }
}

#[test]
fn call_home_equal_opaque_numbers_from_different_origins_remain_distinct() {
    let mut first=Arena::new();let one=origin(id(1),0);let a=result(&mut first,one);
    let mut second=Arena::new();let two=origin(id(2),0);let b=result(&mut second,two);
    assert_eq!(first.op(a),second.op(b),"fixture must collide in local opaque numbering");
    let mut arena=Arena::new();
    let a=arena.graft(&first,a,&mut HashMap::new());
    let b=arena.graft(&second,b,&mut HashMap::new());assert_ne!(a,b);
    let sum=arena.bin(BinOp::Add,a,b);
    let graph=cfg(arena,vec![block(id(1),vec![call(one,a)],Terminator::Jump(id(2))),
        block(id(2),vec![call(two,b)],Terminator::Jump(id(3))),
        block(id(3),Vec::new(),Terminator::Return {dest:sum})]);
    let plan=CallHomePlan::build(&graph);
    assert_eq!(plan.slot_for(one),Some(0));assert_eq!(plan.slot_for(two),Some(1));
}

#[test]
fn call_home_same_origin_grafted_to_multiple_refs_uses_one_home() {
    let mut source=Arena::new();let old=origin(id(1),0);let raw=result(&mut source,old);
    let mut arena=Arena::new();let def=arena.graft(&source,raw,&mut HashMap::new());
    let use_=arena.graft(&source,raw,&mut HashMap::new());assert_ne!(def,use_);
    let graph=cfg(arena,vec![block(id(1),vec![call(old,def)],Terminator::Jump(id(2))),
        block(id(2),Vec::new(),Terminator::Return {dest:use_})]);
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
}

#[test]
fn call_home_visits_memory_boxed_exit_and_terminator_operands() {
    let (mut graph,old,value)=linear();
    let zero=graph.arena.constant(0,Width::W64);
    graph.blocks[1].terminator=Terminator::Return {dest:zero};
    graph.blocks[1].events=vec![
        Event::Store {addr:zero,value,width:Width::W64,site:1,region:Region::VmScratch},
        Event::Load {addr:value,value:zero,width:Width::W64,site:2,region:Region::Guest},
        Event::Boxed {site:3,text:String::new(),bytes:Vec::new(),
            mem:Some(BoxedMem {addr:value,bytes:8,writes:false}),defs:Vec::new(),
            uses:vec![(iced_x86::Register::RAX,value)]},
        Event::StackProbe {site:4,top:value,size:zero},
    ];
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
    graph.blocks[1].events.clear();graph.blocks[1].exit_regs.push((Reg::Rax,value));
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
}

#[test]
fn call_home_owner_identity_includes_vip_and_index() {
    for change in 0..2 {
        let mut arena=Arena::new();let owner=BlockId {handler:1,vip:Some(123)};
        let old=origin(owner,if change==0 {1} else {0});let value=result(&mut arena,old);
        let actual=if change==0 {owner} else {BlockId {handler:1,vip:Some(124)}};
        let graph=cfg(arena,vec![block(actual,vec![call(old,value)],Terminator::Jump(id(2))),
            block(id(2),Vec::new(),Terminator::Return {dest:value})]);
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),None);
    }
}

#[test]
fn call_home_no_crossblock_use_or_unreachable_use_gets_no_slot() {
    let (mut graph,old,value)=linear();graph.blocks.pop();
    graph.blocks[0].terminator=Terminator::Return {dest:value};
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),None);
    graph.blocks.push(block(id(2),Vec::new(),Terminator::Return {dest:value}));
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),None);
}

#[test]
fn review_call_home_dominated_diamond_preserves_one_definition() {
    let (mut graph,old,value)=linear();
    let condition=graph.arena.init_reg(Reg::Rcx);
    graph.blocks[0].terminator=Terminator::Branch {predicate:condition,taken:id(2),not_taken:id(3)};
    graph.blocks[1].terminator=Terminator::Jump(id(4));
    graph.blocks.push(block(id(3),Vec::new(),Terminator::Jump(id(4))));
    graph.blocks.push(block(id(4),Vec::new(),Terminator::Return {dest:value}));
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
}

#[test]
fn review_call_home_use_only_loop_has_one_immutable_definition() {
    let (mut graph,old,value)=linear();
    let condition=graph.arena.init_reg(Reg::Rcx);
    graph.blocks[1].terminator=Terminator::Branch {predicate:condition,taken:id(2),not_taken:id(3)};
    graph.blocks[1].exit_regs.push((Reg::Rbx,value));
    graph.blocks.push(block(id(3),Vec::new(),Terminator::Return {dest:value}));
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
}

#[test]
fn review_call_home_diamond_bypass_and_cyclic_definition_still_reject() {
    for mode in 0..2 {
        let (mut graph,old,value)=linear();
        let condition=graph.arena.init_reg(Reg::Rcx);
        if mode==0 {
            graph.blocks.insert(0,block(id(0),Vec::new(),Terminator::Branch {
                predicate:condition,taken:id(1),not_taken:id(2)}));
            graph.entry=id(0);
        } else {
            graph.blocks[1].terminator=Terminator::Branch {
                predicate:condition,taken:id(1),not_taken:id(3)};
            graph.blocks.push(block(id(3),Vec::new(),Terminator::Return {dest:value}));
        }
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),None,"mode {mode}");
    }
}

#[test]
fn review_call_home_zero_iteration_and_unreachable_predecessor_reject() {
    for dead in [false,true] {
        let (mut graph,old,value)=linear();
        let condition=graph.arena.init_reg(Reg::Rcx);
        let bypass=block(id(0),Vec::new(),Terminator::Branch {
            predicate:condition,taken:id(1),not_taken:id(2)});
        graph.blocks.push(bypass);
        if !dead {
            graph.entry=id(0);
            graph.blocks[0].terminator=Terminator::Jump(id(0));
        }
        // No loop iteration is required to reach the use; with `dead`, the
        // disconnected predecessor remains unknown rather than manufacturing
        // a definition for an otherwise dominated reachable use.
        graph.blocks[1].exit_regs.push((Reg::Rbx,value));
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),None,"dead={dead}");
    }
}

#[test]
fn review_call_home_dead_island_does_not_poison_a_closed_dominated_path() {
    let (mut graph,old,_)=linear();
    graph.blocks.push(block(id(9),Vec::new(),Terminator::Jump(id(9))));
    assert_eq!(CallHomePlan::build(&graph).slot_for(old),Some(0));
}

#[test]
fn review_call_home_unreachable_use_and_duplicate_definition_still_reject() {
    for duplicate in [false,true] {
        let (mut graph,old,value)=linear();
        let events=if duplicate {vec![call(old,value)]} else {Vec::new()};
        graph.blocks.push(block(id(9),events,Terminator::Return {dest:value}));
        assert_eq!(CallHomePlan::build(&graph).slot_for(old),None,"duplicate={duplicate}");
    }
}
