$(document).ready(function(){
    const ipt_change = (ipt)=>{
        var dt = $(ipt).data()
        dt.changed = $(ipt).val()
        return dt
    }
    
    const all_change = ()=>{
        var result =[]
        /*
        var all_ipt = $(".cell_edit_ipt")
        for(var i in all_ipt)
        {result.push(ipt_change(all_ipt[i]))}
        */
        $(".cell_edit_ipt").each(function(){
            result.push(ipt_change(this))
        })
        return result
    }
    window.all_change = all_change
    $("#save_all_change").click(function(){
        console.log(all_change())
    })

    const fetch_data = (page,where)=>{
        const title = $("#table_area").data("title")
        fetch(`/${title}/df_api?page=${page}&where=${where}`)
            .then((res)=>res.text())
            .then((text)=>{
                $("#table_area").html(text)
                assign_cell()
            })
            .catch((error)=>{
                console.log(error)
            })
    }
    window.fetch_data = fetch_data
    fetch_data(0,"")

    const search_btn =  ()=>{
        const page = parseInt($("#page_number").val())-1
        const where = $("#where").val()
        if(page<0){
            alert(`page number '${page}' is not correct`)
            return {}
        }
        fetch_data(page,where)
    }

    $("#where").change(function(){
        $("#page_number").val(1)
    })

    $("#search_btn").click(search_btn)

    $("#last-page").click(function(){
        var page = parseInt($("#page_number").val())
        if(page<=1){
            $("#page_number").val(1)
            return null
        }else{
            $("#page_number").val(page-1)
            search_btn()
        }
    })

    $("#next-page").click(function(){
        var page = parseInt($("#page_number").val())
        $("#page_number").val(page+1)
        search_btn()
    })

    const submit_data = ()=>{
        const title = $("#table_area").data("title")
        const page = parseInt($("#page_number").val())-1
        const where = $("#where").val()
        const data= {changes:all_change(),query:{where,page}}

        fetch(`/${title}/save_api`,{
            method:'POST',
            cache:'no-cache',
            headers:{
                'Content-Type':'application/json'
            },
            body:JSON.stringify(data)
        })
            .then((res)=>res.text())
            .then((text)=>{
                $("#table_area").html(text)
                assign_cell()
            })
            .catch((error)=>{
                console.log(error)
            })
    }
    
    $("#save_all_change").click(submit_data)

    const assign_cell = ()=>{
    $(".data_cell").dblclick(function(){
        var val = $(this).text();
        var td = $(this).parent("td");
        var ipt = document.createElement("input");
        $(ipt).val(val);
        ipt.className="cell_edit_ipt form-control";
        ipt.width=this.width
        var dt= $(this).data();
        $(ipt).data(dt);
        $(this).html("");
        $(td).append(ipt);
    })
    }
    
})
