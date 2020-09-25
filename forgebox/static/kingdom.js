class KingDom{
    constructor(tag,status={}){
        this=document.createElement(tag);
        this.status = status;
        this.actions = []
    }
    add_trigger=(...events)=>{
        for(var i in events){
            var event = events[i];
            event=this.update()
        }
    }
    add_action=(...actions)=>{
        this.actions.push(...actions)
    }
    update=()=>{
        for(var i in actions){
            var action = actions[i];
            action(this)
        }
    }
}