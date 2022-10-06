Search.setIndex({docnames:["convolution","correctness","design","examples","hostcode","index","matrix","observers","tutorial","user-api"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:4,sphinx:56},filenames:["convolution.rst","correctness.rst","design.rst","examples.rst","hostcode.rst","index.rst","matrix.rst","observers.rst","tutorial.ipynb","user-api.rst"],objects:{"kernel_tuner.c":{CFunctions:[2,0,1,""]},"kernel_tuner.c.CFunctions":{benchmark:[2,1,1,""],cleanup_lib:[2,1,1,""],compile:[2,1,1,""],memcpy_dtoh:[2,1,1,""],memset:[2,1,1,""],ready_argument_list:[2,1,1,""],run_kernel:[2,1,1,""]},"kernel_tuner.core":{benchmark:[2,3,1,""],check_kernel_correctness:[2,3,1,""],compile_kernel:[2,3,1,""]},"kernel_tuner.cuda":{CudaFunctions:[2,0,1,""]},"kernel_tuner.cuda.CudaFunctions":{benchmark:[2,1,1,""],compile:[2,1,1,""],copy_constant_memory_args:[2,1,1,""],memcpy_dtoh:[2,1,1,""],memset:[2,1,1,""],ready_argument_list:[2,1,1,""],run_kernel:[2,1,1,""]},"kernel_tuner.opencl":{OpenCLFunctions:[2,0,1,""]},"kernel_tuner.opencl.OpenCLFunctions":{benchmark:[2,1,1,""],compile:[2,1,1,""],memcpy_dtoh:[2,1,1,""],memset:[2,1,1,""],ready_argument_list:[2,1,1,""],run_kernel:[2,1,1,""]},"kernel_tuner.runners":{sequential_brute_force:[2,2,0,"-"]},"kernel_tuner.runners.sequential_brute_force":{run:[2,3,1,""]},"kernel_tuner.util":{detect_language:[2,3,1,""],get_grid_dimensions:[2,3,1,""],get_problem_size:[2,3,1,""],get_thread_block_dimensions:[2,3,1,""],looks_like_a_filename:[2,3,1,""],prepare_kernel_string:[2,3,1,""],prepare_list_of_files:[2,3,1,""],replace_param_occurrences:[2,3,1,""],setup_block_and_grid:[2,3,1,""],setup_kernel_strings:[2,3,1,""]},kernel_tuner:{core:[2,2,0,"-"],run_kernel:[9,3,1,""],tune_kernel:[9,3,1,""],util:[2,2,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","module","Python module"],"3":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:method","2":"py:module","3":"py:function"},terms:{"0":[0,1,2,4,5,6,7,8,9],"000001":9,"018869400024":8,"02665598392":8,"03752319813":8,"04807043076":8,"05":8,"05262081623":8,"054880023":8,"05549435616":8,"05816960335":8,"05957758427":8,"06":[2,9],"0629119873":8,"06332798004":8,"06672639847":8,"06709122658":8,"06844799519":8,"06983039379":8,"07002239227":8,"0731967926":8,"07386879921":8,"07484800816":8,"07508480549":8,"0759360075":8,"0799423933":8,"08220798969":8,"08389122486":8,"09015038013":8,"09730558395":8,"09794559479":8,"0f":8,"0x2aaab952f240":8,"0x2aaabbdcb2e8":8,"0x2aab1c98b3c8":8,"1":[0,1,2,4,7,8,9],"10":[7,8],"1000":8,"10000000":5,"10033922195":8,"10066559315":8,"10125439167":8,"1024":8,"10700161457":8,"10740480423":8,"11":8,"112":0,"11514236927":8,"12":[7,8],"12000002861":8,"12033278942":8,"128":[0,5,8],"128x32":8,"13":8,"13023357391":8,"13297917843":8,"14":[7,8],"14420480728":8,"14729599953":8,"15":[5,8],"15089921951":8,"15916161537":8,"16":[0,1,4,6,8],"17":[0,1,4,8],"18":8,"18713598251":8,"19084160328":8,"1e":[1,2,9],"1e3":[7,8],"1xn":6,"2":[0,1,3,4,5,6,7,8,9],"2000":8,"2017":5,"22305920124":8,"225":8,"225f":8,"256":0,"26":8,"2634239912":8,"29789438248":8,"2d":[0,3,8],"2u_":8,"3":[0,1,4,5,6,7,8,9],"31661438942":8,"32":[0,2,6,8,9],"3269824028":8,"32x2":8,"4":[0,6,7,8],"4096":[0,1,4,6,8],"4164":8,"423038482666016":8,"432":0,"48":[0,8],"4u_":8,"5":[5,7,8],"500":8,"53":8,"538227200508":8,"539891195297":8,"540352010727":8,"540383994579":8,"542387211323":8,"542937588692":8,"544691193104":8,"550105595589":8,"554745602608":8,"560505592823":8,"562521612644":8,"563417613506":8,"565254402161":8,"56585599184":8,"567417597771":8,"568556785583":8,"569388794899":8,"573836791515":8,"575859189034":8,"576044797897":8,"577215993404":8,"578681600094":8,"578745603561":8,"579411196709":8,"579904007912":8,"58035838604":8,"581280004978":8,"588492810726":8,"59088640213":8,"595276796818":8,"597267186642":8,"5cm":7,"6":[0,1,4,6,8,9],"60216319561":8,"605760002136":8,"60942081213":8,"615148806572":8,"618003201485":8,"618598401546":8,"621254396439":8,"622867202759":8,"624492788315":8,"625260794163":8,"626163220406":8,"626976013184":8,"627136015892":8,"631142401695":8,"632006394863":8,"637958395481":8,"638348805904":8,"64":[0,5,6,8],"643359994888":8,"643820810318":8,"646092808247":8,"648620784283":8,"649779188633":8,"64x4":8,"650336003304":8,"652575993538":8,"657920002937":8,"662041604519":8,"662566399574":8,"66344319582":8,"666003203392":8,"666656005383":8,"667251205444":8,"667347204685":8,"673248004913":8,"675232005119":8,"675923216343":8,"676595199108":8,"677363204956":8,"679372787476":8,"680422389507":8,"681350398064":8,"682188808918":8,"685670387745":8,"68781440258":8,"687955200672":8,"689356791973":8,"690009605885":8,"691116797924":8,"691385602951":8,"692665600777":8,"694451200962":8,"69627519846":8,"697094392776":8,"699366402626":8,"7":[2,5,8],"700883197784":8,"70140799284":8,"703302407265":8,"705055999756":8,"705900788307":8,"705932807922":8,"710278391838":8,"713843202591":8,"714169609547":8,"716115188599":8,"7168192029":8,"72":8,"721862399578":8,"722668802738":8,"723999989033":8,"725548803806":8,"726335990429":8,"727967989445":8,"730982398987":8,"731334400177":8,"731891202927":8,"732409596443":8,"733248019218":8,"735436797142":8,"740518403053":8,"741964805126":8,"75":7,"75041918755":8,"750636804104":8,"752479994297":8,"759308815":8,"759679996967":8,"760915207863":8,"761139214039":8,"763775992393":8,"766662418842":8,"768064010143":8,"771103990078":8,"77759360075":8,"779033613205":8,"782060790062":8,"78363519907":8,"788345599174":8,"791257584095":8,"792108798027":8,"792595207691":8,"797900807858":8,"799059200287":8,"8":[0,7,8],"80":0,"801119995117":8,"801798415184":8,"801996803284":8,"803033602238":8,"803718411922":8,"804953610897":8,"805299210548":8,"806828796864":8,"808000004292":8,"808211183548":8,"821881604195":8,"822137594223":8,"824838399887":8,"826515209675":8,"832300806046":8,"833420813084":8,"835481595993":8,"835494399071":8,"837299215794":8,"837804794312":8,"838195204735":8,"840755212307":8,"840908801556":8,"841631996632":8,"843411195278":8,"843692803383":8,"844428789616":8,"848044800758":8,"851040017605":8,"852166390419":8,"852575981617":8,"853574407101":8,"85437438488":8,"85886080265":8,"860332798958":8,"862348806858":8,"867276787758":8,"869497597218":8,"875001597404":8,"876377594471":8,"876627194881":8,"888671982288":8,"890803205967":8,"893279993534":8,"9":[0,1,4,7,8],"900499212742":8,"922745585442":8,"93347837925":8,"96":0,"971545600891":8,"997139203548":8,"999763202667":8,"abstract":[2,7],"boolean":[2,6,9],"byte":2,"case":[2,6,7,8,9],"class":[2,7],"default":[2,7,8,9],"do":[2,4,8,9],"final":[0,8],"float":[0,2,4,5,6,7,8,9],"function":[0,1,2,4,5,7,8,9],"import":[0,1,2,5,7,8],"int":[0,2,5,6,8,9],"long":[4,8],"new":[5,7,8],"public":5,"return":[1,2,4,7,8,9],"true":[0,1,4,6,8,9],"try":[0,6,8,9],"void":[0,5,6,8],"while":[6,7,8],A:[0,2,6,9],And:[6,8],As:[6,7,8],At:[1,2],Be:8,But:8,By:4,For:[1,2,6,7,8,9],If:[0,1,4,5,8,9],In:[1,4,6,7,8,9],It:[2,4,5,7,8,9],On:[8,9],One:8,Or:5,That:[4,8],The:[0,1,2,4,6,7,8,9],Then:[6,8],There:[3,4,5,6,7,8],These:[7,8,9],To:[0,1,4,5,7,8,9],With:[4,5],_:8,__global:5,__global__:[0,5,6,8],__kernel:5,__shared__:[6,8],__synchthread:6,__syncthread:[6,8],_funcptr:2,a100:7,a6000:7,abc:7,abl:[1,2,8],about:[2,5,7,8,9],abov:[0,2,8],absolut:[1,9],abstractmethod:7,access:[2,7,8],accord:9,account:[4,5],accur:[4,7],accuraci:7,across:[2,4,6],act:7,actual:[1,2,6,8],ad:[4,8,9],add:[0,2,6,7,8],addit:[2,5,7,8],addtion:8,advanc:5,advantag:7,affect:8,after:[1,2,4,5,7,8,9],after_finish:7,after_start:7,afterward:5,again:8,against:[1,2],aggreg:7,aim:5,algebra:6,algorithm:[3,6],all:[2,3,4,5,6,7,8,9],allclos:[1,9],alloc:[2,3,4,8,9],allow:[1,2,5,6,7,8,9],almost:[1,7,8],along:2,alreadi:[0,6,8],also:[0,2,4,5,6,7,8,9],although:7,alwai:8,among:[0,8],amount:[0,6,7,8],an:[1,2,3,4,5,6,7,8,9],analysi:8,analyz:8,ani:[2,4,5,6,7,8,9],anoth:[4,7,8],answer:[1,2,3,5,8,9],api:[2,5],appli:8,applic:[3,4,7,8],approach:7,appropi:8,approx:8,approxim:8,ar:[0,1,2,3,4,5,6,7,8,9],arch:8,architectur:[2,7],area:[6,8],aren:6,arg:[0,1,4,5,6,8],argument:[1,2,3,4,5,6,8,9],arithmet:[8,9],around:[0,3],arrai:[0,2,8,9],artifact:2,assum:[0,8],assumpt:8,astyp:[0,1,4,5,6,8],atol:[1,2,9],attempt:2,author:5,auto:[5,7],autom:5,automat:[8,9],avail:[3,5,7,8],averag:[2,4,7,8],avoid:0,ax1:8,ax2:8,axesimag:8,b:[5,6],back:[4,9],backend:[2,4,7],bandwidth:6,bank:6,base:[2,7,9],basic:[2,8],becaus:[4,6,7,8],becom:8,been:[4,8],befor:[1,4,5,6,7,8,9],before_start:7,begin:[7,8],behavior:7,behind:4,being:[1,2,7,8],below:[3,4],ben:5,benchmark:[1,2,3,4,5,7,8,9],benchmarkobserv:7,benvanwerkhoven:5,berend:5,best:[0,5,8],best_tim:8,better:[5,8],between:[4,5,7,8,9],beyond:[2,8,9],biologi:8,bit:[2,4,6,8],block:[0,2,3,5,6,7,8,9],block_siz:6,block_size_i:[0,1,4,6,8,9],block_size_str:8,block_size_x:[0,1,4,5,6,8,9],block_size_z:[0,8,9],blockdim:[0,9],blockidx:[0,5,6,8],boilerpl:8,border:4,both:[3,5,6,8],bottom:2,bound:6,boundari:8,branch:5,brute:[2,7],buffer:2,build:[2,5,8],built:[0,7,9],bulk:8,burtscher_measuring_2014:7,bx:8,c:[3,4,5,6,9],c_arg:2,cach:8,call:[0,2,4,5,6,7,8,9],can:[0,2,3,4,5,6,7,8,9],cannot:8,cap:7,capabl:[6,8,9],caption:7,card:6,care:8,caus:[7,8],cc:8,cd:5,cedric:5,cell:8,center:[3,7],central:8,certain:[7,8],chang:[5,7,9],changelog:5,check:[1,2,4,8],check_kernel_correct:2,chemistri:8,choic:4,choos:[6,8],chunk:4,cite:[5,7],clarifi:4,clean:3,cleaner:8,cleanup:8,cleanup_lib:2,clock:7,clone:[5,8],close:8,closer:8,cltune:5,cluster:2,cmem_arg:[1,2,9],code:[0,2,3,6,9],collabor:6,collect:[7,8],color:8,column:6,columnwidth:7,com:5,combin:[0,2,3,5,6,8,9],come:[7,8],command:5,common:[5,7],commonli:8,commun:8,compar:[1,7,8],comparison:1,compat:5,compil:[0,1,2,3,4,5,7,8,9],compile_kernel:2,compiler_opt:[2,9],complet:5,complex:[4,6,8],compos:7,comput:[0,1,2,3,4,6,9],compute_capability_major:8,compute_capability_minor:8,concentr:8,concept:[2,8],condens:8,condit:8,configur:[0,2,3,5,6,7,8,9],confus:0,connect:7,conserv:6,consid:[2,6],constant:[2,3,4,8,9],construct:1,consumpt:[6,7],contain:[2,4,5,6,8,9],content:5,context:8,continu:[7,8],control:[1,7,8],conveni:[2,4,8],convent:[2,4,9],convert:8,convolut:[1,4,5,6],convolution_correct:1,convolution_kernel:[0,1],convolution_na:1,convolution_stream:[4,5],cooler:8,copi:[2,5,8,9],copy_constant_memory_arg:2,core:7,correct:[4,9],correspond:[0,6,8],cost:8,could:[2,4,6,7,8,9],counter:7,counterpart:2,cours:8,cover:8,creat:[0,2,5,6,7,8],creation:2,crucial:7,csv:[3,8],ctype:2,cu:[0,1,4],cuda:[0,1,3,4,6,8,9],cudamemcpytosymbol:4,cudastreamwaitev:4,cupi:7,current:[0,1,2,5,7,8,9],current_modul:7,current_problem_s:2,custom:7,czarnul:7,d:8,d_filter:1,data:[2,4,7,8,9],datafram:8,de:5,decreas:0,def:[7,8],defin:[2,6,7,8],degrad:8,degre:8,delta:8,demonstr:[1,6],depend:[3,7],deriv:[7,8],descret:8,describ:[4,7],design:[5,7,8],dest:2,detail:[2,5,9],detect:[2,9],detect_languag:2,determin:[2,7,8],dev:[2,7],develop:[1,2],devic:[2,3,4,5,7,8,9],devicealloc:2,devprop:8,df:8,dict:[0,1,2,4,5,6,7,9],dictionari:[0,2,5,8,9],did:8,differ:[0,1,2,3,4,6,7,8,9],difficult:8,diffus:5,diffuse_kernel:8,dim:2,dimens:[2,3,4,5,7,8,9],dimension:[3,9],direct:[0,2,4,6,7,8,9],directli:[2,4,5,7,8],directori:[0,4,5,8],discard:2,discontinu:6,discuss:[2,8],distanc:8,distant:8,distribut:[2,6],divid:[0,4,8,9],divison:9,divisor:[2,8,9],doc:5,docstr:5,document:[7,8],doe:[2,4,7,8,9],doesn:5,domain:[3,8],don:[4,8,9],done:5,doubl:8,down:6,downsid:7,dramat:5,drastic:6,driver:[2,8],drv:8,dt:8,due:[6,9],dump:8,durat:7,dure:[2,7,8,9],e:7,each:[0,1,2,6,7,8,9],earlier:8,easi:[7,8,9],easili:[7,8],effect:[6,8],effici:7,either:2,element:[6,7,8,9],em:7,empti:9,en:7,enabl:7,end:[0,7,8],energi:7,enough:6,ensur:[1,4,5,7,8],enter:8,entir:[2,8],entri:[5,8],environ:5,equal:[8,9],equat:8,equi:8,error:[1,4],estim:8,evalu:[6,8],even:[0,4,5,6,8],event:[4,7,8],eventu:2,everi:[0,1,3,5,7,8],everyth:8,everywher:8,exact:5,exactli:[2,7,8],exampl:[1,4,7,8,9],except:[2,3],exchang:8,execut:[2,3,4,5,6,7,8,9],exhaust:7,expand:7,expect:[0,1,2,5,6,8,9],explain:[4,6,8],explan:5,expos:2,express:[3,4,6,8,9],extend:7,extens:[2,5],f:[0,1,4],facilit:7,fact:[4,6,7,8],factor:[0,5,6,8],fail:9,fals:9,far:[0,5,8],fast:[1,6,8],faster:8,fastest:2,featur:[1,3,5,7,9],feel:8,few:[4,8],fewer:8,field:[1,8],field_copi:8,fig:[7,8],figur:7,file:[2,3,4,8,9],filenam:[2,3,9],filipovivc2021us:7,fill:[2,6],filter:[0,1,3,4],filter_height:0,filter_width:0,find:[0,4,5],fine:8,finish:[4,7],first:[2,4,6,7,8,9],fit:4,fix:8,flexibl:[2,6,8],float32:[0,1,2,4,5,6,8,9],fly:8,follow:[0,2,4,5,7,8,9],forc:[2,7],fork:5,form:[6,7],format:[2,8],formula:8,fortran:5,fortun:6,forward:6,four:8,fp:8,frac:8,free:[4,8],frequenc:7,frequent:6,friendli:7,from:[0,1,2,3,4,5,7,8,9],full:5,func:[2,7],further:[0,6,8],g:7,gcc:2,geforc:8,gemm:7,gener:[5,8],get:[0,5,8],get_attribut:8,get_devic:8,get_funct:8,get_global_id:5,get_grid_dimens:2,get_initial_condit:8,get_kernel_str:8,get_local_s:9,get_problem_s:2,get_result:7,get_thread_block_dimens:2,gflop:7,gh:5,git:5,github:[5,8],give:[0,8],given:[0,5,7,8,9],global:[2,8],go:8,goal:5,goe:6,good:[1,8],googl:5,got:8,gpu:[0,2,3,4,5,6,7,9],gpu_arg:2,gpu_result:8,gradual:7,grain:8,graph:7,graphic:7,great:8,grid:[2,3,4,8,9],grid_div_i:[0,1,2,4,6,8,9],grid_div_x:[0,1,2,4,6,8,9],grid_div_z:[2,9],grid_size_i:4,grid_size_x:4,group:[2,8,9],grow:8,gt:8,gtx:8,guess:8,guid:7,h:4,ha:[4,7,8,9],half:8,halt:4,hand:6,handl:[4,9],hardwar:[7,8],have:[0,2,4,5,6,7,8,9],heat:8,height:0,here:[4,5,6],high:[6,7,8],highest:2,highli:[5,6],hit:7,hold:[0,8,9],homepag:8,hook:7,host:[2,3,7,9],hot:8,hotspot:8,how:[0,1,3,4,6,8,9],howev:[0,1,4,6,7,8,9],html:5,http:[5,7],hz:7,i:[0,1,4,5,6,8],idea:[0,4,8],ignor:[2,8,9],illustr:3,imag:[0,7,8],image_height:0,image_width:0,impact:[4,8],implement:[1,2,3,6,7],impli:5,importantli:[6,8],improv:8,imshow:8,includ:[0,1,2,4,5,8,9],includegraph:7,increas:[0,7,8],independ:[5,6],index:[5,6],inform:[2,7,8],init:8,initi:[2,8],inlin:8,input:[0,1,3,4,5,7,8,9],input_s:[0,1,4],input_width:0,insert:[1,2,4],insid:[4,5,8,9],inspect:7,instal:[4,8],instanc:[1,2,4,8,9],instance_str:2,instant:8,instantan:7,instead:[3,6,9],instruct:[3,6,8],int32:[0,2,5,9],integ:9,intel:4,intend:5,interact:7,intercept:7,interest:[1,6],interfac:[1,2,4,7,9],intermedi:8,intern:2,interv:7,intricaci:7,introduc:[0,5,7,8],io:7,item:[2,8],iter:[2,6,7,8],its:[1,2,7,8,9],itself:4,j:[0,6,8],java:5,join:8,json:[3,8],jump:7,jupyt:8,just:[2,4,5,6,8],k:[6,8],kb:6,keep:8,kei:[2,8,9],kernel:[2,4,6,7,9],kernel_file_list:2,kernel_nam:[2,4,9],kernel_str:[0,1,2,4,5,6,8,9],kernel_tun:[0,1,4,6,8,9],khz:7,know:[6,7,8],known:6,krzywaniak:7,krzywaniak_performanceenergy_2019:7,l1:8,l2:8,label:7,lambda:[7,8],lang:[2,3,4,9],languag:[2,4,9],larg:[8,9],larger:[4,8],later:[8,9],latter:4,launch:[0,2,4,7,8,9],layer:[2,7],learn:[2,5],least:5,left:[2,6,7,8],let:[0,5,6,8],level:[5,7],librari:[2,5,7],like:[2,5,8,9],likewis:8,limit:[3,6,7,8,9],line:8,linear:6,list:[0,1,2,3,4,5,6,7,8,9],littl:[2,8],ll:[6,8],load:2,locat:1,lock:3,log:9,longer:7,look:[4,8],looks_like_a_filenam:2,lookup:2,loop:[6,8],loss:8,lot:[5,6,8,9],low:[7,8],lower:7,lowest:2,lt:8,machin:[2,5],mai:[1,2,4,5,6,7,8,9],main:[2,5,7],maintain:2,make:[5,6,7,8],make_context:8,manag:[7,8],mandatori:7,mani:[0,2,5,6,7,8,9],manipul:2,map:3,match:[0,1,2,5],mathema:5,matmul_kernel:6,matplotlib:8,matrix:[5,7],matter:[4,5,8],maxim:7,maximum:[1,9],md:5,mean:[4,6,7],meantim:5,measur:[2,4,7,8,9],melt:8,mem:2,mem_alloc:8,memcpi:[2,4],memcpy_dtoh:[2,8],memcpy_htod:8,memori:[2,3,4,6,7,9],memset:2,mention:7,merg:8,messi:8,metal:8,method:[4,7,8],metric:[2,7],might:7,millisecond:8,min:8,mind:8,minim:[4,7],minimalist:8,misc:5,mix:2,mode:7,model:8,modifi:1,modul:[2,4,5,7],moment:[1,2,8],more:[0,2,5,6,7,8,9],most:[2,3,4,6,7,8],mostli:[2,9],motion:8,move:[2,4,6,8,9],ms:8,much:[6,7,8,9],multipl:[4,6,7,9],multipli:5,multiprocessor:8,must:9,n:[4,5,6,8],naiv:[0,1,6,8],name:[2,5,6,7,8,9],nbyte:8,ndarrai:[0,2],ndrang:2,nearest:9,necessari:[8,9],necessarili:4,need:[1,2,4,5,6,7,8,9],next:[6,8],nice:8,node:2,non:1,none:[1,2,5,9],noodl:2,normal:9,nosetest:[3,5],note:[0,6,7,8,9],notebook:8,notic:8,notifi:1,now:[0,2,4,6,8],nugteren:5,num_reg:7,num_stream:4,num_thread:9,number:[0,2,3,5,6,7,8,9],numer:8,numpi:[0,1,2,4,5,6,8,9],nvcc:2,nvidia:7,nvml:7,nvml_gr_clock:7,nvml_mem_clock:7,nvml_pwr_limit:7,nvml_test:7,nvmlobserv:7,nx:8,ny:8,object:[0,2,7,8,9],observ:7,obtain:[7,8],occup:6,occur:[2,7],occurr:[2,6],offici:7,often:[7,8],onc:[1,2,7,8,9],one:[0,2,5,6,7,8,9],ones:8,onli:[1,3,4,5,6,7,8,9],open:[0,1,4,6,8],opencl:[3,4,8,9],oper:[4,6,7,8],opportun:6,oppos:7,optim:[1,4,5,6,7,8],option:[1,3,4,5,6,8,9],order:[0,1,2,4,5,7,8],ordereddict:[7,8],origin:[0,2,5],original_kernel:[2,9],other:[0,2,4,5,7,8],our:[0,7,8],out:[0,1,6],output:[0,1,3,4,5,6,8,9],over:[2,6,7,8],overhead:8,overlap:[3,4,5],overwritten:7,own:[2,4,7],p:7,page:[3,5,8],pair:8,panda:[3,8],paper:5,parallel:[2,8],param:[1,2,9],paramet:[1,2,3,4,5,7,8,9],parameter_spac:2,pars:8,part:[6,7,8,9],partial:8,particular:[2,4,7,8],particularli:7,pass:[0,1,2,3,4,5,6,7,8,9],pattern:7,pcie:7,per:[0,7,8,9],perform:[0,1,2,4,5,6,7,8],physic:8,pick:6,pip:5,pixel:0,place:[7,8],plai:8,plain:4,plan:2,platform:[2,5,9],pleas:[1,5],plot:8,png:7,point:[2,4,7,8],pointer:2,pop:8,possibl:[0,2,4,5,7,8,9],power:[6,7],powerful:5,powersensor2:7,powersensor:7,powersensorobserv:7,pragma:[6,8],precis:4,prefer:8,prepar:[2,5,8],prepare_kernel_str:2,prepare_list_of_fil:2,prepend:2,preprocessor:2,press:8,previou:8,previous:[2,6,8],print:8,privileg:7,problem:[0,2,3,4,5,6,8,9],problem_s:[0,1,2,4,6,8,9],proce:6,process:[0,2,5,6,7,8],processor:4,prod:[0,1,4,6],produc:[1,5],product:[8,9],program:[0,1,4,5,8],programm:7,project:5,promis:7,provid:[2,4,5,8,9],prune:2,ps_energi:7,pull:5,purpos:[0,4,6,8,9],put:8,py:[1,4],pybind11:7,pybind11footnot:7,pycuda:[2,4,5,7,8],pyopencl:[2,5,7],pypi:5,pyplot:8,pytest:3,python:[2,3,4,6,7,9],qualiti:[2,4],quantiti:[7,8],queue:5,quick:8,quickli:8,quit:8,r:[0,1,4],radiat:8,ramp:7,randint:8,randn:[0,1,4,5,6],random:[0,1,2,4,5,6,8],rang:[0,1,4,5,6,8],rather:[6,7,8,9],re:8,read:[0,1,2,4,7,8,9],readi:[2,8],readthedoc:7,ready_argument_list:2,realiz:6,realli:8,reason:[0,2,6,9],recent:[2,7],record:[7,8],redistribut:8,reduc:[6,8],reduct:[7,9],redund:6,ref:7,refer:[0,3,4,7,8,9],reflect:7,refresh:7,region:8,regist:[6,7,8],registerobserv:7,regular:[2,7],rel:5,relev:2,rememb:[0,6,8],repeatedli:[2,7],replac:[0,1,2,5,6,8,9],replace_param_occurr:2,report:[7,9],repositori:[0,5,8],repres:[2,8],request:[5,7,9],requir:[4,5,7,8],resolut:7,resourc:9,respect:[0,7],respons:[2,6],rest:[2,8],restart:8,restrict:[3,6,9],result:[1,2,5,6,7,9],reus:8,right:[7,8],robust:2,romein_powersensor_2018:7,root:7,round:[8,9],row:6,rtx:7,run:[1,2,3,4,5,7,8,9],run_kernel:[1,2,3,9],runtim:[7,8,9],s:[0,4,6,7,8,9],sa:6,sai:[0,2,8],same:[0,1,2,4,5,8,9],sampl:[2,9],satisfi:9,save:8,sb:6,scalar:[2,8,9],scientif:5,scipi:3,script:[2,5,6],scriptsiz:7,search:[2,3,5,6,7,9],second:[7,8],secondli:6,section:[2,7,8],see:[2,4,5,7,8,9],seem:[0,6,8],seen:[2,8],select:[7,8,9],self:7,sensor:7,separ:[2,3,4],sequenti:2,seri:2,serv:[7,8],set:[0,1,2,3,7,8,9],set_titl:8,setup:[4,8],setup_block_and_grid:2,setup_kernel_str:2,sever:[3,5,6,8,9],sh_u:8,share:[2,6,9],sheet:8,shift:8,should:[0,1,2,4,5,6,7,8,9],show:[3,5,6,7,8],shown:[0,2,7],shuffl:3,significantli:7,similar:[2,4,6,9],simpl:[2,3,4,7,8],simpli:[2,6,8],simplic:8,simplifi:8,sinc:0,singl:[0,1,2,4,6,8,9],singular:9,size:[0,1,2,3,4,5,6,8,9],skip:[8,9],slightli:4,slowest:2,sm_:8,small:8,so:[0,2,4,5,6,7,8],softwar:[5,7,8],solut:7,solv:6,some:[2,5,6,7,8,9],somehow:8,someth:8,sometim:8,sourc:[2,4,5],sourcemodul:8,space:[1,2,4,6,7,9],spatial:8,special:8,specif:[2,3,6,7,8,9],specifi:[0,1,2,4,7,8,9],spent:[4,5],sphinxdoc:5,split:4,spread:4,squar:6,src:2,stabil:7,stabl:7,start:[1,4,6,7,8],state:[2,7,8,9],statement:2,stdout:8,step:[6,7,8],still:[1,5,6],store:[0,2,6],str:8,strategi:[2,7],stream:[5,8],string:[2,3,5,6,8,9],structur:[6,8],style:5,subplot:8,subscrib:7,subsect:7,subsubsect:7,suffer:7,suffici:7,sum:[0,6],summar:9,suppli:[0,4,9],support:[2,4,5,7,8,9],suppos:[0,8],sure:[0,5,8],symbol:[2,9],synchron:[7,8],system:7,t:[4,5,6,8,9],tad:6,take:[0,2,7,8],tdp:7,techniqu:6,tell:[0,3,4,8],temperatur:[7,8],temporari:2,term:[0,7],terminolog:8,test:[3,5,7,8,9],test_vector_add:3,text:8,texttt:7,than:[2,5,6,7,8,9],thei:[2,3,7,8,9],them:[0,2,4,5],therefor:[0,1,2,4,6,8],thi:[0,1,2,4,5,6,7,8,9],thin:7,thing:[4,5],think:[2,8],those:3,thousand:8,thread:[0,2,3,5,7,8,9],threadidx:[0,5,6,8],three:[2,7],through:[0,2,7,8],ti:8,tician:5,tile:[3,5,6],tile_size_i:[0,1,4,6,8,9],tile_size_x:[0,1,4,6,8],time:[0,2,4,5,7,8,9],time_sinc:8,timer:4,titan:[7,8],titl:5,tj:8,to_csv:8,togeth:[8,9],toler:1,too:[4,6,8,9],took:8,tool:5,toolkit:5,top:[2,5,7],total:[6,7,8],total_flop:7,track:7,transfer:[3,4,5],transmit:7,treat:9,tri:8,trust:1,tunabl:[2,3,5,7,8,9],tune:[1,2,3,7,9],tune_kernel:[0,1,2,4,5,6,8,9],tune_param:[0,1,2,4,5,6,8,9],tuner:[0,1,2,4,7,9],tupl:[2,9],tutori:5,two:[0,2,6,8,9],tx:[6,8],ty:[6,8],type:[0,1,2,3,5,7,8,9],typic:[2,6,9],u:8,u_:8,u_new:8,u_old:8,unawar:0,undefin:8,under:7,underutil:8,unit:[2,5],unload:2,unrol:[5,6,8],until:[4,7],up:[7,8,9],upgrad:5,upload:4,us:[0,1,2,3,4,5,6,7,9],usag:[6,7],usb:7,use_noodl:9,user:[2,7],usual:[6,7],v:8,valid:[3,6],valu:[0,1,2,3,4,6,7,8,9],van:5,vari:[6,7,8],variabl:[0,2,6,9],variou:[5,7],ve:8,vector:8,vector_add:[4,5],verbos:[0,1,2,4,6,8,9],veri:[1,4,5,6,7,8],verifi:[1,3,5,9],voltag:7,vspace:7,w:[7,8],wa:[2,7,8,9],wai:[1,4,6,7,8,9],want:[1,4,5,6,9],warp:9,warpsiz:6,we:[0,1,2,3,4,6,7,8],weel:5,welcom:8,well:[6,8],went:8,were:[6,8,9],werkhoven:5,what:[0,1,2,4,7,8,9],whatev:[2,4],when:[2,4,6,7,8,9],whenev:1,where:[6,7,8,9],whether:[2,5,7,9],which:[0,2,3,4,5,6,7,8,9],whole:8,whose:[1,9],why:[4,8],wide:[0,8],width:[0,6,7],wish:2,within:[0,2,6,8,9],without:[4,5,7,8],work:[0,2,7,8,9],worri:8,worst:8,would:[0,8],wrapper:7,write:[3,9],written:5,x:[0,5,6,8,9],x_:8,x_i:8,xyz:9,y:[0,4,6,8,9],y_:8,year:5,yet:[4,6],yield:6,you:[0,1,2,3,4,5,6,8,9],your:[3,4,5,8],yourself:[4,9],z:9,zero:[0,1,4],zeros_lik:[5,6,8]},titles:["Convolution Example","Kernel Correctness Verification","Design documentation","Kernel Tuner Examples","Tuning Host Code","The kernel_tuner documentation","Matrix Multiply Example","&lt;no title&gt;","Kernel Tuner Tutorial - Diffusion","API Documentation"],titleterms:{A:5,The:5,add:3,all:0,api:9,argument:0,auto:8,c:2,cfunction:2,citat:5,code:[4,5,8],comput:8,contribut:5,convolut:[0,3],convolution_correct:3,convolution_stream:3,core:2,correct:[1,5],cuda:[2,5],cudafunct:2,data:6,depend:5,design:2,diffus:8,dimens:0,document:[2,5,9],exampl:[0,3,5,6],gpu:8,grid:0,guid:5,host:[4,5],implement:8,increas:6,indic:5,instal:5,introduct:5,kernel:[0,1,3,5,8],kernel_tun:[2,5],matrix:[3,6],memori:8,multipl:3,multipli:6,number:4,opencl:[2,5],openclfunct:2,paramet:[0,6],per:6,point:3,polygon:3,put:0,py:3,python:[5,8],reduct:3,relat:5,result:8,reus:6,runner:2,sepconv:3,sequential_brute_forc:2,setup:[0,6],share:8,simpl:5,spars:3,stencil:3,store:8,stream:4,tabl:5,thread:6,tile:8,togeth:0,tune:[0,4,5,6,8],tuner:[3,5,8],tutori:8,us:8,usag:5,util:2,vector:3,verif:[1,5],work:[5,6]}})