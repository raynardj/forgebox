var mlm_visualize = (data,text)=>{
    var token_data = restructure(data,text)
    console.log(data)
    console.log(display_sentence(token_data))
}
var display_sentence=(token_data)=>{
        seq_token = token_data[0];
        var sentence='';
        for(token in seq_token){
            sentence+=token.render_token();
        }
        return `<span>${sentence}</span>`
}

var attention_factory=(attention)=>{
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
            return this.mask.map(i=>i.slice(seq_idx,seq_idx+1))
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
    return {Attention,Layer,AttentionHead}
}

var token_factory=(text,attention)=>{
    var Token = class{
        constructor(data){
            this.seq_idx = data.seq_idx;
            this.x = data.x;
            this.token = data.token;
            this.is_mask = data.is_mask;
        }
        text=text;
        attention=attention;
    }
    get_mask=(layer,head)=>{
        return this.attention
        .layers[layer]
        .heads[head]
    }
    get_mask_forward=(layer,head)=>{
        return this.get_mask(layer,head)
            .get_mask_forward(this.seq_idx)
    }
    get_mask_backward=(layer,head)=>{
        return this.get_mask(layer,head)
            .get_mask_backward(this.seq_idx)
    }
    render_token=()=>{
        return `<span data-seq_idx=${this.seq_dix}
        data-x=${this.x}
        data-is_mask='${this.is_mask}'
        >${this.token}</span>`
    }
    return Token
}

var restructure=(data,text)=>{
    var {
        x,mask,pred_idx,
        attention,pred_tokens,mapper
    } = data
    var {AttentionHead,Layer,Attention} = attention_factory(attention)
    
    var att_obj= new Attention()
    var Token = token_factory(text,att_obj)
    var new_data = {}
    mapper=mapper[0]
    for(var m=0;m<pred_tokens.length;m++){
        /* Iter through mask tokens*/
        var the_mask = {}
        for(var i=0;i<x[0].length;i++){
            /* Iter through the senquence length*/
            var the_token = new Token({
                seq_idx:i,
                x:x[0][i],
                token:text.slice(mapper[i][0],mapper[i][1]),
                is_mask:mask[0][i],
            })
            the_mask[i] = the_token
        } 
        new_data[m] = the_mask
    }
    return new_data
}