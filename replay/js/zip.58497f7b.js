(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["zip"],{"9d25":function(t,e,n){"use strict";n.d(e,"b",(function(){return i})),n.d(e,"d",(function(){return r})),n.d(e,"c",(function(){return o})),n.d(e,"a",(function(){return l})),n.d(e,"f",(function(){return c})),n.d(e,"e",(function(){return s}));var a=n("b32d"),i={id:0,status:"draft",title:"",fullContent:"",abstractContent:"",sourceURL:"",imageURL:"",timestamp:"",platforms:["a-platform"],disableComment:!1,importance:0,author:"",reviewer:"",type:"",pageviews:0},r=function(t){return Object(a["a"])({url:"/articles",method:"get",params:t})},o=function(t,e){return Object(a["a"])({url:"/articles/".concat(t),method:"get",params:e})},l=function(t){return Object(a["a"])({url:"/articles",method:"post",data:t})},c=function(t,e){return Object(a["a"])({url:"/articles/".concat(t),method:"put",data:e})},s=function(t){return Object(a["a"])({url:"/pageviews",method:"get",params:t})}},ca54:function(t,e,n){"use strict";n.r(e);var a=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"app-container"},[n("el-input",{staticStyle:{width:"300px"},attrs:{placeholder:"Please enter the file name (default file)","prefix-icon":"el-icon-document"},model:{value:t.filename,callback:function(e){t.filename=e},expression:"filename"}}),n("el-button",{staticStyle:{"margin-bottom":"20px"},attrs:{loading:t.downloadLoading,type:"primary",icon:"el-icon-document"},on:{click:t.handleDownload}},[t._v(" Export Zip ")]),n("el-table",{directives:[{name:"loading",rawName:"v-loading",value:t.listLoading,expression:"listLoading"}],attrs:{data:t.list,"element-loading-text":"Loading...",border:"",fit:"","highlight-current-row":""}},[n("el-table-column",{attrs:{align:"center",label:"ID",width:"95"},scopedSlots:t._u([{key:"default",fn:function(e){var n=e.$index;return[t._v(" "+t._s(n)+" ")]}}])}),n("el-table-column",{attrs:{label:"Title"},scopedSlots:t._u([{key:"default",fn:function(e){var n=e.row;return[t._v(" "+t._s(n.title)+" ")]}}])}),n("el-table-column",{attrs:{label:"Author",align:"center",width:"180"},scopedSlots:t._u([{key:"default",fn:function(e){var a=e.row;return[n("el-tag",[t._v(t._s(a.author))])]}}])}),n("el-table-column",{attrs:{label:"Readings",align:"center",width:"115"},scopedSlots:t._u([{key:"default",fn:function(e){var n=e.row;return[t._v(" "+t._s(n.pageviews)+" ")]}}])}),n("el-table-column",{attrs:{label:"Date",align:"center",width:"220"},scopedSlots:t._u([{key:"default",fn:function(e){var a=e.row;return[n("i",{staticClass:"el-icon-time"}),n("span",[t._v(t._s(a.timestamp))])]}}])})],1)],1)},i=[],r=(n("96cf"),n("1da1")),o=n("d4ec"),l=n("bee2"),c=n("262e"),s=n("2caf"),u=n("9ab4"),d=n("1b40"),f=n("9d25"),p=n("d257"),m=(n("4160"),n("d3b7"),n("25f0"),n("159b"),n("21a6")),h=n("c4e3"),b=n.n(h),g=function(t,e){var n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:"file",a=arguments.length>3&&void 0!==arguments[3]?arguments[3]:"file",i=new b.a,r=e,o="".concat(t,"\r\n");r.forEach((function(t){var e="";e=t.toString(),o+="".concat(e,"\r\n")})),i.file("".concat(n,".txt"),o),i.generateAsync({type:"blob"}).then((function(t){Object(m["saveAs"])(t,"".concat(a,".zip"))}),(function(t){alert("Zip export failed: "+t.message)}))},v=function(t){Object(c["a"])(n,t);var e=Object(s["a"])(n);function n(){var t;return Object(o["a"])(this,n),t=e.apply(this,arguments),t.list=[],t.listLoading=!0,t.downloadLoading=!1,t.filename="",t}return Object(l["a"])(n,[{key:"created",value:function(){this.fetchData()}},{key:"fetchData",value:function(){var t=Object(r["a"])(regeneratorRuntime.mark((function t(){var e,n,a=this;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return this.listLoading=!0,t.next=3,Object(f["d"])({});case 3:e=t.sent,n=e.data,this.list=n.items,setTimeout((function(){a.listLoading=!1}),500);case 7:case"end":return t.stop()}}),t,this)})));function e(){return t.apply(this,arguments)}return e}()},{key:"handleDownload",value:function(){this.downloadLoading=!0;var t=["Id","Title","Author","Readings","Date"],e=["id","title","author","pageviews","timestamp"],n=this.list,a=Object(p["b"])(e,n);""!==this.filename?g(t,a,this.filename,this.filename):g(t,a),this.downloadLoading=!1}}]),n}(d["c"]);v=Object(u["a"])([Object(d["a"])({name:"ExportZip"})],v);var w=v,_=w,y=n("2877"),j=Object(y["a"])(_,a,i,!1,null,null,null);e["default"]=j.exports}}]);