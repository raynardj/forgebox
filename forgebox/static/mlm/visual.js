var mlm_visualize = (data, text, output_id) => {
    var {
        Sentence,
        ControlBar,
        att_obj
    } = attention_factory(data, text, output_id)
    var sentence = new Sentence();
    var control_bar = new ControlBar({
        nb_heads: att_obj.layers[0].heads.length,
        current_head: 0
    });

    control_bar.sensitive.add_callback(
        control_bar.func_print_change,
        control_bar.change_current_head(att_obj),
    )

    $(`#${output_id}`).append(control_bar.dom)
    $(`#${output_id}`).append(sentence.display())

    window.sentence = sentence
    window.control_bar = control_bar
}

var text_span=(text)=>{
    var dom = document.createElement("span");
    $(dom).html(text)
    return dom
}

var SensitiveDOM = class {
    constructor(status, render_how) {
        this.status = status;
        this.render_how = render_how;
        this.change_hows = [];
    }
    render = () => {
        this.dom = this.render_how(this.status)
        return this.dom
    }
    callback = () => {
        for (var i = 0; i < this.change_hows.length; i++) {
            this.change_hows[i](this.dom, this.status);
        }
    }
    add_callback = (...change_hows) => {
        for(var i=0;i<change_hows.length;i++){
            this.change_hows.push(change_hows[i]);
        }
    }
}

var argmax = (arr) => {
    /*
    The position of the max number
    */
    var result = 0;
    var max_ = arr[0];
    for (var i = 0; i < arr.length; i++) {
        if (arr[i] > max_) {
            result = i;
            max_ = arr[i];
        }
    }
    return result
}
var threshold_filter = (arr, threshold = .05) => {
    /*
    filter the array to satisfy a threshold for each float number
    the length of the array will probably shrink
    */
    var result = [];
    for (var i = 0; i < arr.length; i++) {
        if (arr[i] > threshold) {
            result.push(i);
        }
    }
    return result
}

var coloring = (color) => {
    var coloring_function = (tk) => {
        $(tk.dom).css({
            "color": color
        })
    };
    return coloring_function
}
var shading = (color, px) => {
    var shading_function = (tk) => {
        $(tk.dom).css({
            "text-shadow": `0px 0px ${px}px ${color}`
        })
    };
    return shading_function
}

var attention_factory = (data, text, output_id) => {
    var {
        x,
        mask,
        pred_idx,
        attention,
        pred_tokens,
        mapper
    } = data

    var AttentionHead = class {
        /*
            Such class manages an sequence length x sequence length
             of attention mask
            */
        constructor(head) {
            this.mask = head
        }
        get_mask_forward(seq_idx) {
            return this.mask[seq_idx]
        }
        get_mask_backward(seq_idx) {
            return this.mask.map(i => i[seq_idx])
        }
    }

    var Layer = class {
        constructor(layer) {
            this.layer = layer
            this.heads = this.build_heads()
        }
        build_heads() {
            var heads = []
            for (var h = 0; h < this.layer.length; h++) {
                var new_head = new AttentionHead(this.layer[h]);
                new_head.head_idx = h;
                heads.push(new_head);
            }
            return heads
        }
    }

    var Attention = class {
        attention = attention;
        current_head=0;
        constructor() {
            this.layers = this.build_layers()
        }
        build_layers() {
            var layers = []
            for (var l = 0; l < this.attention.length; l++) {
                var new_layer = new Layer(this.attention[l][0])
                new_layer.layer_idx = l
                layers.push(new_layer)
            }
            return layers
        }
    }

    var att_obj = new Attention();

    var Token = class {
        constructor(data) {
            this.seq_idx = data.seq_idx;
            this.x = data.x;
            this.start = data.start;
            this.end = data.end;
            this.is_mask = data.is_mask;
            this.sentence = data.sentence;

            this.token = text.slice(this.start, this.end)
        }
        text = text;
        attention = att_obj;
        nb_layers = att_obj.layers.length;

        back = (i, head, cb = null, th_cb = null) => {
            if (i < this.nb_layers) {
                var layer = this.nb_layers - 1 - i;
                var mask_slice = this.get_mask_backward(layer, head)
                var next_idx = argmax(mask_slice)

                var next_token = this.sentence.tokens[next_idx]
                if (th_cb != null) {
                    var th_idx_arr = threshold_filter(mask_slice, );
                    var th_tokens = th_idx_arr.map(i => this.sentence.tokens[i]);

                    th_tokens.map(th_token => th_cb(th_token))

                }
                if (cb != null) {
                    cb(next_token);
                }

                next_token.back(i + 1, head, cb, th_cb)
            }
        }

        get_mask = (layer, head) => {
            return this.attention
                .layers[layer]
                .heads[head]
        };
        get_mask_forward = (layer, head) => {
            return this.get_mask(layer, head)
                .get_mask_forward(this.seq_idx)
        };
        get_mask_backward = (layer, head) => {
            return this.get_mask(layer, head)
                .get_mask_backward(this.seq_idx)
        };
        assign_click = (dom) => {
            var onclick = () => {
                this.sentence.recover()
                this.back(0, this.attention.current_head, coloring("red"), shading("blue", 5))
            }
            dom.onclick = onclick;
        }
        render_token = (last_pos = null) => {
            var token_dom = document.createElement("span");
            this.dom = token_dom;

            this.tangible_dom = document.createElement("span")
            $(this.tangible_dom).addClass('mlm-vis-token')
            $(this.tangible_dom).html(this.token)

            this.intangible_dom = null;
            if (last_pos !== null) {
                if (this.start > last_pos) {
                    this.intangible_dom = document.createElement("span");
                    $(this.intangible_dom).addClass('mlm-vis-token-niche');

                    var niche = this.text.slice(last_pos, this.start)
                        .replace(/\n/gi, "<br>")

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
                seq_idx: this.seq_idx,
                x: this.x,
                is_mask: this.is_mask,
                start: this.start,
                end: this.end,
            })
            this.assign_click(token_dom);
            return {
                token_dom,
                last_pos,
                niche
            }
        }
    }
    var Sentence = class {
        constructor() {
            console.log({text,mapper})
            this.text = text;
            this.mapper = mapper;
            this.x = x;
            this.attention = att_obj;

            this.create_sentence()
        }
        

        create_sentence() {
            this.tokens = {}
            for (var i = 0; i < this.x[0].length; i++) {
                this.tokens[i] = new Token({
                    seq_idx: i,
                    x: this.x[0][i],
                    is_mask: mask[0][i],
                    start: mapper[i][0],
                    end: mapper[i][1],
                    sentence: this,
                })
            }
        }
        display = () => {
            /*
            return the sentence dom
            */
            var sentence_dom = document.createElement("div");
            var last_pos = 0;
            for (var i in this.tokens) {

                var {
                    token_dom,
                    last_pos
                } = this.tokens[i]
                    .render_token(last_pos);
                $(sentence_dom).append(token_dom);
            }
            this.dom = sentence_dom
            return sentence_dom
        }
        map_token = (callback) => {
            /*
            apply a function to all token
            */
            for (var t in this.tokens) {
                var token = this.tokens[t];
                callback(token);
            }
        }
        recover = () => {
            /*
            Recover the intial states of the tokens
            */
            var color_func = coloring("#000");
            var shading_func = shading("rgba(255,255,255,0)", 0)

            this.map_token(color_func);
            this.map_token(shading_func);
        }
        underline_on = () => {
            this.map_token((tk) => {
                $(tk.tangible_dom).css({
                    'text-decoration': 'underline'
                })
            })
        }
        underline_off = () => {
            this.map_token((tk) => {
                $(tk.tangible_dom).css({
                    'text-decoration': null
                })
            })
        }
    }
    var ControlBar = class {
        constructor(data) {
            this.sensitive = new SensitiveDOM(data, this.create_dom())
            this.sensitive.render()
            this.dom = this.sensitive.dom
            
        }
        func_print_change=(dom,data)=>{
            console.log(`current_head: ${data.current_head}`);
            $(dom.title_idx_dom).html(data.current_head);
        }
        change_current_head=(attention)=>{
            return (dom,data)=>{
                attention.current_head = data.current_head
            }
        }

        create_dom() {
            var cbar = this;
            var render_how = (data) => {
                /* "this" means a Sensitive DOM*/
                var {
                    nb_heads,
                    current_head
                } = data
                this.dom = document.createElement("div")

                this.title_dom = document.createElement("h3");
                this.dom.title_idx_dom = text_span(current_head);
                $(this.title_dom).append(text_span("Current Head: "))
                $(this.title_dom).append(this.dom.title_idx_dom)
                $(this.dom).append(this.title_dom)

                this.btn_group = document.createElement("div")
                $(this.btn_group).addClass("btn").addClass("btn-group")
                $(this.dom).append(this.btn_group)

                this.dom.btn_list = []
                for (var h = 0; h < nb_heads; h++) {
                    var btn = document.createElement("span");
                    $(btn).addClass("btn")
                        .addClass("btn-success")
                        .addClass("btn-xs")
                    this.dom.btn_list.push(btn)
                    $(btn).html(`H ${h}`)
                    $(btn).data({head: h})
                    $(btn).click(function () {
                        cbar.sensitive.status.current_head=$(this).data("head");
                        cbar.sensitive.callback()
                    })
                    $(this.btn_group).append(btn)
                }
                return this.dom
            }
            return render_how
        }
    }

    return {
        Attention,
        Layer,
        AttentionHead,
        Sentence,
        ControlBar,
        Token,
        att_obj
    }
}