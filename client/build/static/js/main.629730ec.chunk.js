(this.webpackJsonpclient=this.webpackJsonpclient||[]).push([[0],{48:function(e,t,a){e.exports=a(78)},53:function(e,t,a){},54:function(e,t,a){},78:function(e,t,a){"use strict";a.r(t);var n=a(0),i=a.n(n),r=a(9),c=a.n(r),l=(a(53),a(12)),s=a(13),m=a(16),o=a(15),u=(a(54),a(95)),h=a(100),d=a(97),g=a(99),E=a(7),p=a.n(E),v=a(46),f=a(98),_=function(e){Object(m.a)(a,e);var t=Object(o.a)(a);function a(){return Object(l.a)(this,a),t.apply(this,arguments)}return Object(s.a)(a,[{key:"render",value:function(){return i.a.createElement(u.a,{container:!0,spacing:0},i.a.createElement(u.a,{item:!0,md:12},i.a.createElement(d.a,{gutterBottom:!0,variant:"h4"}," Thank you. Your baby is in safe hands!!!")),i.a.createElement(u.a,{item:!0,md:12},i.a.createElement("img",{src:"PROFILE.png",height:"50%",width:"40%"})),i.a.createElement(u.a,{item:!0,md:6},i.a.createElement("img",{src:"ultra0.png",height:"50%",width:"50%"})),i.a.createElement(u.a,{item:!0,md:6},i.a.createElement("img",{src:"ultra1.png",height:"50%",width:"50%"})),i.a.createElement(u.a,{item:!0,md:6},i.a.createElement("img",{src:"ultra2.png",height:"50%",width:"50%"})),i.a.createElement(u.a,{item:!0,md:6},i.a.createElement("img",{src:"ultra3.png",height:"50%",width:"50%"})))}}]),a}(n.Component),y=a(10),b=a.n(y),k=function(e){Object(m.a)(a,e);var t=Object(o.a)(a);function a(){return Object(l.a)(this,a),t.apply(this,arguments)}return Object(s.a)(a,[{key:"componentWillMount",value:function(){var e=this;this.setState({start:!1,image_count:0,ultra_image_count:-1,doneAnalysis:!1,ultra_images:[]}),setInterval((function(){b.a.get("patient").then((function(t){-1!=t.data.ultra_loc&&e.setState({image_count:t.data.ultra_loc})}))}),1e3),setInterval((function(){b.a.get("doneAnalysis").then((function(t){e.setState({doneAnalysis:t.data.doneAnalysis})}))}),1e3)}},{key:"handleStart",value:function(){this.setState({start:!0})}},{key:"incrementCount",value:function(){b.a.post("patient",{patient_img_count:this.state.image_count}),this.state.ultra_images.includes(this.state.image_count)||this.setState({ultra_images:this.state.ultra_images.concat(this.state.image_count)})}},{key:"render",value:function(){var e=this;return i.a.createElement(u.a,{container:!0},this.state.start?i.a.createElement(u.a,{item:!0,md:12,className:"App-header",justify:"center",alignItems:"center"},i.a.createElement(h.a,{direction:"up",in:this.state.start},i.a.createElement(_,null))):i.a.createElement(u.a,{item:!0,md:12},i.a.createElement(h.a,{direction:"down",in:!this.state.start},i.a.createElement("header",{className:"App-header"},i.a.createElement("img",{src:"ultrasound_position".concat(this.state.image_count,".png"),height:"50%",width:"40%"}),i.a.createElement("br",null),i.a.createElement(u.a,{container:!0},0!==this.state.ultra_images.size&&this.state.ultra_images.map((function(e,t){return i.a.createElement(u.a,{item:!0,md:3},i.a.createElement("img",{src:"ultra".concat(e,".png"),height:"60%",width:"60%"}))}))),this.state.doneAnalysis?i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handleStart()}},"View Results \xa0",i.a.createElement(p.a,null)):i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.incrementCount()}},"Done Scan \xa0",i.a.createElement(p.a,null),i.a.createElement(p.a,null))))))}}]),a}(n.Component),w=function(e){Object(m.a)(a,e);var t=Object(o.a)(a);function a(){return Object(l.a)(this,a),t.apply(this,arguments)}return Object(s.a)(a,[{key:"componentWillMount",value:function(){var e=this;this.setState({start:!1,image_count:0,ultra_image_count:-1,doneAnalysis:!1,ultra_images:[],ultra_label:[]}),setInterval((function(){b.a.get("physician").then((function(t){-1!==t.data.patient_img&&(e.setState({ultra_image_count:t.data.patient_img}),-1===t.data.patient_img||e.state.ultra_images.includes(t.data.patient_img)||e.setState({ultra_images:e.state.ultra_images.concat(t.data.patient_img),ultra_label:e.state.ultra_label.concat(t.data.img_label)}))}))}),1e3),b.a.get("patient").then((function(t){-1!=t.data.ultra_loc&&e.setState({image_count:t.data.ultra_loc})}))}},{key:"handleMove",value:function(e){b.a.post("physician",{image_count:e}),this.setState({image_count:e})}},{key:"incrementCount",value:function(){this.setState({image_count:this.state.image_count+1})}},{key:"handleDone",value:function(){b.a.post("doneAnalysis",{doneAnalysis:!0})}},{key:"render",value:function(){var e=this;return i.a.createElement(u.a,{container:!0},this.state.start?i.a.createElement(u.a,{item:!0,md:12,className:"App-header",justify:"center",alignItems:"center"},i.a.createElement(h.a,{direction:"up",in:this.state.start},i.a.createElement(_,null))):i.a.createElement(u.a,{item:!0,md:12},i.a.createElement(h.a,{direction:"down",in:!this.state.start},i.a.createElement("header",{className:"App-header"},i.a.createElement("img",{src:"ultrasound_position".concat(this.state.image_count,".png"),height:"50%",width:"40%"}),i.a.createElement("br",null),i.a.createElement("div",{style:{width:"100%"}},i.a.createElement(u.a,{container:!0},0!==this.state.ultra_images.size&&this.state.ultra_images.map((function(t,a){return i.a.createElement(u.a,{item:!0,md:3},i.a.createElement("img",{src:"ultra".concat(t,".png"),height:"40%",width:"40%"}),i.a.createElement(d.a,{variant:"h5",component:"h6"},e.state.ultra_label[a]))}))),i.a.createElement(u.a,{container:!0,spacing:0},i.a.createElement(u.a,{item:!0,md:3},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handleMove(1)}},"Move Left \xa0",i.a.createElement(p.a,null))),i.a.createElement(u.a,{item:!0,md:3},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handleMove(3)}},"Move Up \xa0",i.a.createElement(p.a,null))),i.a.createElement(u.a,{item:!0,md:3},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handleMove(0)}},"Move Down \xa0",i.a.createElement(p.a,null))),i.a.createElement(u.a,{item:!0,md:3},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handleMove(2)}},"Move Right \xa0",i.a.createElement(p.a,null))),i.a.createElement(u.a,{item:!0,md:12},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handleDone()}},"Done \xa0",i.a.createElement(p.a,null)))))))))}}]),a}(n.Component),S=Object(v.a)({typography:{fontFamily:"'Montserrat', sans-serif"}}),j=function(e){Object(m.a)(a,e);var t=Object(o.a)(a);function a(){return Object(l.a)(this,a),t.apply(this,arguments)}return Object(s.a)(a,[{key:"componentWillMount",value:function(){this.setState({startPatient:!1,startPhysician:!1}),b.a.get("Splash")}},{key:"handlePatient",value:function(){this.setState({startPatient:!0})}},{key:"handlePhysician",value:function(){this.setState({startPhysician:!0})}},{key:"render",value:function(){var e=this;return i.a.createElement(f.a,{theme:S},i.a.createElement("div",{className:"App"},i.a.createElement(u.a,{container:!0},this.state.startPatient||this.state.startPhysician?i.a.createElement(u.a,{item:!0,md:12,className:"App-header",justify:"center",alignItems:"center"},this.state.startPatient?i.a.createElement(k,null):i.a.createElement(w,null)):i.a.createElement(u.a,{item:!0,md:12},i.a.createElement(h.a,{direction:"down",in:!this.state.start},i.a.createElement("header",{className:"App-header"},i.a.createElement("img",{src:"logo_back.png",width:"20%",height:"20%"}),i.a.createElement(d.a,{gutterBottom:!0,variant:"h3"},"Welcome to the UltraAssist!!!"),i.a.createElement(u.a,{container:!0,spacing:2},i.a.createElement(u.a,{item:!0,md:6},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handlePhysician()}},"Start as Physician\xa0",i.a.createElement(p.a,null))),i.a.createElement(u.a,{item:!0,md:6},i.a.createElement(g.a,{variant:"extended",onClick:function(){return e.handlePatient()}},"Start as Patient\xa0",i.a.createElement(p.a,null))))))))))}}]),a}(n.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));c.a.render(i.a.createElement(i.a.StrictMode,null,i.a.createElement(j,null)),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))}},[[48,1,2]]]);
//# sourceMappingURL=main.629730ec.chunk.js.map