var mlm_visualize = (data,text,output_id)=>{
    var {Sentence,} = attention_factory(data,text,output_id)
    
    var sentence = new Sentence()

    $(`#${output_id}`).append(sentence.display({with_heads:true}))
    window.sentence = sentence
}
var Status=class{
    constructor(data){
        this={...this,...data};
    }
}
var get_theme=(idx)=>{
    var mapping = {
        0:"primary",
        1:"success",
        2:"info",
        3:"warning",
        4:"danger",
    }
    return mapping[idx%5]
}

var argmax = (arr)=>{
    var result = 0;
    var max_ = arr[0];
    for(var i=0;i<arr.length;i++){
        if(arr[i]>max_){
            result=i;
            max_=arr[i];
        }
    }
    return result
}
var threshold_filter = (arr,threshold=.05)=>{
    var result = [];
    for(var i=0;i<arr.length;i++){
        if(arr[i]>threshold){
            result.push(i);
        }
    }
    return result
}

var coloring = (color)=>{
    var coloring_function = (tk)=>{
    $(tk.dom).css({"color":color})
    };
    return coloring_function
}
var shading = (color,px)=>{
    var shading_function = (tk)=>{
        $(tk.dom).css({"text-shadow":`0px 0px ${px}px ${color}`})
    };
    return shading_function
}

var attention_factory=(data,text,output_id)=>{
    var {
        x,mask,pred_idx,
        attention,pred_tokens,mapper
    } = data

    var AttentionHead= class{
        /*
            Such class manages an sequence length x sequence length
             of attention mask
            */
        constructor(head){
            this.mask = head
        }
        get_mask_forward(seq_idx){
            return this.mask[seq_idx]
        }
        get_mask_backward(seq_idx){
            return this.mask.map(i=>i[seq_idx])
        }
    }
    
    var Layer = class{
        constructor(layer){
            this.layer=layer
            this.heads = this.build_heads()
        }
        build_heads(){
            var heads = []
            for(var h=0;h<this.layer.length;h++){
                var new_head = new AttentionHead(this.layer[h]);
                new_head.head_idx = h; 
                heads.push(new_head);
            }
            return heads
        }   
    }

    var Attention =class{
        attention = attention
        constructor(){
            this.layers = this.build_layers()
        }
        build_layers(){
            var layers = []
            for(var l=0;l<this.attention.length;l++){
                var new_layer = new Layer(this.attention[l][0])
                new_layer.layer_idx = l
                layers.push(new_layer)
            }
            return layers
        }
    }

    var att_obj = new Attention();
    var nb_layers=att_obj.layers.length;
    var nb_heads=att_obj.layers[0].heads.length;
    var HeadBtnDOM=class{
        attention = att_obj;
        constructor(data){
            this.head_idx = data.idx;
            this.control = data.control;
            this.status = this.control.status
        }
        click_func=()=>{
            this.status.head_idx = this.head_idx;
        }
        render=()=>{
            this.dom = document.createElement("span");
            this.dom.object = this;
            $(this.dom)
                .addClass("btn")
                .addClass("btn-xs")
                .addClass(`btn-${get_theme(this.head_idx)}`);
            $(this.dom).html(`H ${this.head_idx}`);
            $(this.dom).click(this.click_func)
            return this.dom
        }
    }
    var BtnControl=class{
        constructor(single,iterable){
            this.single=single;
            this.iterable=iterable;
            this.components = []
            this.status = {}
        }
        render = ()=>{
            for(var i=0;i<this.iterable.length;i++){
                var dt = {idx:i,control:this,element:this.iterable[i]}
                this.components.push(new this.single(dt))
            }

            this.dom = document.createElement("div")
            $(this.dom).addClass("btn-group")
            this.dom.object = this

            for(var i=0;i<this.components.length;i++){
                var component = this.components[i].render();
                $(this.dom).append(component);
            }
            return this.dom
        }
        
    }

    var Token = class{
        constructor(data){
            this.seq_idx = data.seq_idx;
            this.x = data.x;
            this.start = data.start;
            this.end = data.end;
            this.is_mask = data.is_mask;
            this.sentence = data.sentence;

            this.token = text.slice(this.start,this.end)
        }
        text=text;
        attention=att_obj;
        

        back=(i,head,cb=null,th_cb=null)=>{
            if(i<nb_layers)
            {
                var layer = nb_layers-1-i;
                var mask_slice = this.get_mask_backward(layer,head)
                var next_idx = argmax(mask_slice)
                
                var next_token = this.sentence.tokens[next_idx]
                if(th_cb!=null){
                    var th_idx_arr = threshold_filter(mask_slice,);
                    var th_tokens = th_idx_arr.map(i=>this.sentence.tokens[i]);

                    th_tokens.map(th_token=>th_cb(th_token))

                }
                if(cb!=null){cb(next_token);}
                
                next_token.back(i+1,head,cb,th_cb)
            }
        }

        get_mask=(layer,head)=>{
            return this.attention
            .layers[layer]
            .heads[head]
        };
        get_mask_forward=(layer,head)=>{
            return this.get_mask(layer,head)
                .get_mask_forward(this.seq_idx)
        };
        get_mask_backward=(layer,head)=>{
            return this.get_mask(layer,head)
                .get_mask_backward(this.seq_idx)
        };
        assign_click=(dom)=>{
            var onclick = ()=>{
                this.sentence.recover()
                this.back(0,0,coloring("red"),shading("rgba(50,60,255,.8)",5))
            }
            dom.onclick = onclick;
        }
        render_token=(last_pos=null)=>{
            var token_dom = document.createElement("span");
            this.dom = token_dom;

            this.tangible_dom = document.createElement("span")
            $(this.tangible_dom).addClass('mlm-vis-token')
            $(this.tangible_dom).html(this.token)

            this.intangible_dom = null;
            if(last_pos!==null){
                if(this.start>last_pos){
                    this.intangible_dom = document.createElement("span");
                    $(this.intangible_dom).addClass('mlm-vis-token-niche');

                    var niche = this.text.slice(last_pos,this.start)
                                    .replaceAll("\n","<br>")

                    $(this.intangible_dom).html(niche);
                    $(this.dom).append(this.intangible_dom);
                    // console.log(`last pos ${last_pos}, niche: '${niche}',this start ${this.start},new_pos ${this.end},`)
                }
            }
            
            last_pos = this.end;
            
            token_dom.id = `token_${this.seq_idx}_${this.start}_${this.end}`;

            $(this.dom).append(this.tangible_dom);

            $(token_dom).addClass("mlm-vis-token-phrase");
            $(token_dom).data({
                seq_idx:this.seq_idx,
                x:this.x,
                is_mask:this.is_mask,
                start:this.start,
                end:this.end,
            })
            this.assign_click(token_dom);
            return {token_dom,last_pos,niche}
        }
    }
    var Sentence = class{
        constructor(){
            this.text=text;
            this.mapper = mapper;
            this.x = x;
            this.attention=att_obj;

            this.create_sentence()
        }
        create_sentence(){
            this.tokens = {}
            for(var i=0;i<this.x[0].length;i++){
                this.tokens[i] = new Token({
                    seq_idx:i,
                    x:x[0][i],
                    is_mask:mask[0][i],
                    start:mapper[i][0],
                    end:mapper[i][1],
                    sentence:this,
                })
            }
        }
        display=(config)=>{
            /*
            return the sentence dom
            */
            this.dom=document.createElement("div");
            var {with_heads}=config;

            $(this.dom).append("<hr>")
            
            if(with_heads){
                this.heads_control = new BtnControl(HeadBtnDOM,[...Array(nb_heads).keys()]);
                $(this.dom).append(this.heads_control.render());
                $(this.dom).append("<hr>")
            }

            var last_pos=0;
            for(var i in this.tokens){
                var {token_dom,last_pos} =this.tokens[i]
                    .render_token(last_pos);
                $(this.dom).append(token_dom);
            }
            return this.dom
        }
        map_token=(callback)=>{
            /*
            apply a function to all token
            */
            for(var t in this.tokens){
                var token = this.tokens[t];
                callback(token);
            }
        }
        recover=()=>{
            /*
            Recover the intial states of the tokens
            */
            var color_func = coloring("#000");
            var shading_func = shading("rgba(255,255,255,0)",0)

            this.map_token(color_func);
            this.map_token(shading_func);
        }
        underline_on = ()=>{
            this.map_token((tk)=>{
                $(tk.tangible_dom).css({'text-decoration': 'underline'})
            })
        }
        underline_off = ()=>{
            this.map_token((tk)=>{
                $(tk.tangible_dom).css({'text-decoration': null})
            })
        }
    }
    

    return {Attention,Layer,AttentionHead,Sentence,Token,att_obj}
}