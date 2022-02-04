Search.setIndex({docnames:["contributing","convolution","correctness","design","diffusion","examples","hostcode","index","install","matrix_multiplication","templates","user-api","vocabulary"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:4,sphinx:56},filenames:["contributing.rst","convolution.ipynb","correctness.rst","design.rst","diffusion.ipynb","examples.rst","hostcode.rst","index.rst","install.rst","matrix_multiplication.ipynb","templates.rst","user-api.rst","vocabulary.rst"],objects:{"kernel_tuner.c":[[3,0,1,"","CFunctions"]],"kernel_tuner.c.CFunctions":[[3,1,1,"","__init__"],[3,1,1,"","benchmark"],[3,1,1,"","cleanup_lib"],[3,1,1,"","compile"],[3,1,1,"","memcpy_dtoh"],[3,1,1,"","memcpy_htod"],[3,1,1,"","memset"],[3,1,1,"","ready_argument_list"],[3,1,1,"","run_kernel"]],"kernel_tuner.core":[[3,0,1,"","DeviceInterface"]],"kernel_tuner.core.DeviceInterface":[[3,1,1,"","__init__"],[3,1,1,"","benchmark"],[3,1,1,"","check_kernel_output"],[3,1,1,"","compile_and_benchmark"],[3,1,1,"","compile_kernel"],[3,1,1,"","copy_constant_memory_args"],[3,1,1,"","copy_shared_memory_args"],[3,1,1,"","copy_texture_memory_args"],[3,1,1,"","create_kernel_instance"],[3,1,1,"","get_environment"],[3,1,1,"","memcpy_dtoh"],[3,1,1,"","ready_argument_list"],[3,1,1,"","run_kernel"]],"kernel_tuner.cuda":[[3,0,1,"","CudaFunctions"]],"kernel_tuner.cuda.CudaFunctions":[[3,1,1,"","__init__"],[3,1,1,"","benchmark"],[3,1,1,"","compile"],[3,1,1,"","copy_constant_memory_args"],[3,1,1,"","copy_shared_memory_args"],[3,1,1,"","copy_texture_memory_args"],[3,1,1,"","memcpy_dtoh"],[3,1,1,"","memcpy_htod"],[3,1,1,"","memset"],[3,1,1,"","ready_argument_list"],[3,1,1,"","run_kernel"]],"kernel_tuner.cupy":[[3,0,1,"","CupyFunctions"]],"kernel_tuner.cupy.CupyFunctions":[[3,1,1,"","__init__"],[3,1,1,"","benchmark"],[3,1,1,"","compile"],[3,1,1,"","copy_constant_memory_args"],[3,1,1,"","copy_shared_memory_args"],[3,1,1,"","copy_texture_memory_args"],[3,1,1,"","memcpy_dtoh"],[3,1,1,"","memcpy_htod"],[3,1,1,"","memset"],[3,1,1,"","ready_argument_list"],[3,1,1,"","run_kernel"]],"kernel_tuner.opencl":[[3,0,1,"","OpenCLFunctions"]],"kernel_tuner.opencl.OpenCLFunctions":[[3,1,1,"","__init__"],[3,1,1,"","benchmark"],[3,1,1,"","compile"],[3,1,1,"","memcpy_dtoh"],[3,1,1,"","memcpy_htod"],[3,1,1,"","memset"],[3,1,1,"","ready_argument_list"],[3,1,1,"","run_kernel"]],"kernel_tuner.runners.sequential":[[3,0,1,"","SequentialRunner"]],"kernel_tuner.runners.sequential.SequentialRunner":[[3,1,1,"","__init__"],[3,1,1,"","run"]],"kernel_tuner.runners.simulation":[[3,0,1,"","SimulationRunner"]],"kernel_tuner.runners.simulation.SimulationRunner":[[3,1,1,"","__init__"],[3,1,1,"","run"]],"kernel_tuner.strategies":[[3,3,0,"-","basinhopping"],[3,3,0,"-","brute_force"],[3,3,0,"-","diff_evo"],[3,3,0,"-","firefly_algorithm"],[3,3,0,"-","genetic_algorithm"],[3,3,0,"-","minimize"],[3,3,0,"-","pso"],[3,3,0,"-","random_sample"],[3,3,0,"-","simulated_annealing"]],"kernel_tuner.strategies.basinhopping":[[3,2,1,"","tune"]],"kernel_tuner.strategies.brute_force":[[3,2,1,"","tune"]],"kernel_tuner.strategies.diff_evo":[[3,2,1,"","tune"]],"kernel_tuner.strategies.firefly_algorithm":[[3,0,1,"","Firefly"],[3,2,1,"","tune"]],"kernel_tuner.strategies.firefly_algorithm.Firefly":[[3,1,1,"","compute_intensity"],[3,1,1,"","distance_to"],[3,1,1,"","move_towards"]],"kernel_tuner.strategies.genetic_algorithm":[[3,2,1,"","disruptive_uniform_crossover"],[3,2,1,"","mutate"],[3,2,1,"","random_population"],[3,2,1,"","random_val"],[3,2,1,"","single_point_crossover"],[3,2,1,"","tune"],[3,2,1,"","two_point_crossover"],[3,2,1,"","uniform_crossover"],[3,2,1,"","weighted_choice"]],"kernel_tuner.strategies.minimize":[[3,2,1,"","get_bounds"],[3,2,1,"","get_bounds_x0_eps"],[3,2,1,"","setup_method_arguments"],[3,2,1,"","setup_method_options"],[3,2,1,"","snap_to_nearest_config"],[3,2,1,"","tune"],[3,2,1,"","unscale_and_snap_to_nearest"]],"kernel_tuner.strategies.pso":[[3,2,1,"","tune"]],"kernel_tuner.strategies.random_sample":[[3,2,1,"","tune"]],"kernel_tuner.strategies.simulated_annealing":[[3,2,1,"","acceptance_prob"],[3,2,1,"","neighbor"],[3,2,1,"","tune"]],"kernel_tuner.util":[[3,4,1,"","SkippableFailure"],[3,2,1,"","check_argument_list"],[3,2,1,"","check_argument_type"],[3,2,1,"","check_restrictions"],[3,2,1,"","check_tune_params_list"],[3,2,1,"","config_valid"],[3,2,1,"","delete_temp_file"],[3,2,1,"","detect_language"],[3,2,1,"","dump_cache"],[3,2,1,"","get_config_string"],[3,2,1,"","get_grid_dimensions"],[3,2,1,"","get_instance_string"],[3,2,1,"","get_kernel_string"],[3,2,1,"","get_number_of_valid_configs"],[3,2,1,"","get_problem_size"],[3,2,1,"","get_smem_args"],[3,2,1,"","get_temp_filename"],[3,2,1,"","get_thread_block_dimensions"],[3,2,1,"","looks_like_a_filename"],[3,2,1,"","normalize_verify_function"],[3,2,1,"","prepare_kernel_string"],[3,2,1,"","print_config_output"],[3,2,1,"","process_cache"],[3,2,1,"","process_metrics"],[3,2,1,"","read_cache"],[3,2,1,"","read_file"],[3,2,1,"","replace_param_occurrences"],[3,2,1,"","setup_block_and_grid"],[3,2,1,"","store_cache"],[3,2,1,"","write_file"]],kernel_tuner:[[11,2,1,"","create_device_targets"],[11,2,1,"","run_kernel"],[11,2,1,"","store_results"],[11,2,1,"","tune_kernel"],[3,3,0,"-","util"]]},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","function","Python function"],"3":["py","module","Python module"],"4":["py","exception","Python exception"]},objtypes:{"0":"py:class","1":"py:method","2":"py:function","3":"py:module","4":"py:exception"},terms:{"0":[1,2,3,4,6,7,9,11],"000001":11,"001":11,"004":7,"018869400024":4,"02665598392":4,"03752319813":4,"04807043076":4,"05":4,"05262081623":4,"054880023":4,"05549435616":4,"05816960335":4,"05957758427":4,"06":11,"0629119873":4,"06332798004":4,"06672639847":4,"06709122658":4,"06844799519":4,"06983039379":4,"07002239227":4,"0731967926":4,"07386879921":4,"07484800816":4,"07508480549":4,"0759360075":4,"0799423933":4,"08":7,"08220798969":4,"08389122486":4,"09015038013":4,"09730558395":4,"09794559479":4,"0f":4,"0x2aaab952f240":4,"0x2aaabbdcb2e8":4,"0x2aab1c98b3c8":4,"1":[1,2,3,4,6,9,11],"10":[4,7,11],"100":11,"1000":4,"10000":3,"1000000":10,"10000000":7,"10033922195":4,"10066559315":4,"10125439167":4,"1016":7,"1024":[1,4],"10700161457":4,"10740480423":4,"11":4,"11514236927":4,"12":4,"12000002861":4,"12033278942":4,"128":[1,4,7,10],"128x32":4,"13":4,"13023357391":4,"13297917843":4,"134233":3,"14":4,"14420480728":4,"14729599953":4,"14991":7,"15":[4,10],"15089921951":4,"15916161537":4,"16":[1,2,4,6,9],"17":[1,2,4,6],"18":4,"18713598251":4,"19084160328":4,"1e":[2,11],"1e3":[1,4,9],"1e9":[1,9],"1xn":9,"2":[1,2,3,4,5,6,9,11],"20":11,"2000":4,"2018":7,"2019":7,"2021":7,"2048":1,"2111":7,"22305920124":4,"225":4,"225f":4,"234342":3,"256":7,"26":4,"2634239912":4,"29789438248":4,"2d":[4,5],"2u_":4,"3":[0,2,4,6,8,9,11],"31661438942":4,"32":[1,3,4,7,9,11],"3269824028":4,"32x2":4,"347":7,"358":7,"4":[1,4,9],"4096":[1,2,4,6,9],"4164":4,"42":1,"423038482666016":4,"48":4,"4u_":4,"5":[0,4,11],"50":11,"500":4,"512":7,"53":4,"538227200508":4,"539891195297":4,"540352010727":4,"540383994579":4,"542387211323":4,"542937588692":4,"544691193104":4,"550105595589":4,"554745602608":4,"560505592823":4,"562521612644":4,"563417613506":4,"565254402161":4,"56585599184":4,"567417597771":4,"568556785583":4,"569388794899":4,"573836791515":4,"575859189034":4,"576044797897":4,"577215993404":4,"578681600094":4,"578745603561":4,"579411196709":4,"579904007912":4,"58035838604":4,"581280004978":4,"588492810726":4,"59088640213":4,"595276796818":4,"597267186642":4,"6":[2,4,6,8,11],"60216319561":4,"605760002136":4,"60942081213":4,"615148806572":4,"618003201485":4,"618598401546":4,"621254396439":4,"622867202759":4,"624492788315":4,"625260794163":4,"626163220406":4,"626976013184":4,"627136015892":4,"631142401695":4,"632006394863":4,"637958395481":4,"638348805904":4,"64":[1,4,7,9,10],"643359994888":4,"643820810318":4,"646092808247":4,"648620784283":4,"649779188633":4,"64x4":4,"650336003304":4,"652575993538":4,"657920002937":4,"662041604519":4,"662566399574":4,"66344319582":4,"666003203392":4,"666656005383":4,"667251205444":4,"667347204685":4,"673248004913":4,"675232005119":4,"675923216343":4,"676595199108":4,"677363204956":4,"679372787476":4,"680422389507":4,"681350398064":4,"682188808918":4,"685670387745":4,"68781440258":4,"687955200672":4,"689356791973":4,"690009605885":4,"691116797924":4,"691385602951":4,"692665600777":4,"694451200962":4,"69627519846":4,"697094392776":4,"699366402626":4,"7":[3,4,11],"700883197784":4,"70140799284":4,"703302407265":4,"705055999756":4,"705900788307":4,"705932807922":4,"710278391838":4,"713843202591":4,"714169609547":4,"716115188599":4,"7168192029":4,"72":4,"721862399578":4,"722668802738":4,"723999989033":4,"725548803806":4,"726335990429":4,"727967989445":4,"730982398987":4,"731334400177":4,"731891202927":4,"732409596443":4,"733248019218":4,"735436797142":4,"740518403053":4,"741964805126":4,"75041918755":4,"750636804104":4,"752479994297":4,"759308815":4,"759679996967":4,"760915207863":4,"761139214039":4,"763775992393":4,"766662418842":4,"768064010143":4,"771103990078":4,"77759360075":4,"779033613205":4,"782060790062":4,"78363519907":4,"788345599174":4,"791257584095":4,"792108798027":4,"792595207691":4,"797900807858":4,"799059200287":4,"8":[1,3,4,9],"801119995117":4,"801798415184":4,"801996803284":4,"803033602238":4,"803718411922":4,"804953610897":4,"805299210548":4,"806828796864":4,"808000004292":4,"808211183548":4,"821881604195":4,"822137594223":4,"824838399887":4,"826515209675":4,"832300806046":4,"833420813084":4,"835481595993":4,"835494399071":4,"837299215794":4,"837804794312":4,"838195204735":4,"840755212307":4,"840908801556":4,"841631996632":4,"843411195278":4,"843692803383":4,"844428789616":4,"848044800758":4,"851040017605":4,"852166390419":4,"852575981617":4,"853574407101":4,"85437438488":4,"85886080265":4,"860332798958":4,"862348806858":4,"867276787758":4,"869497597218":4,"875001597404":4,"876377594471":4,"876627194881":4,"888671982288":4,"890803205967":4,"893279993534":4,"9":[1,2,4,6,11],"90":7,"900499212742":4,"922745585442":4,"93347837925":4,"971545600891":4,"997139203548":4,"999763202667":4,"abstract":3,"boolean":11,"break":[0,10],"byte":[3,11],"case":[1,2,3,4,9,11],"class":3,"default":[1,2,3,4,7,9,10,11],"do":[0,1,3,4,6,9,11],"final":[1,2,4],"float":[1,3,4,6,7,9,10,11],"function":[0,1,2,4,5,6,9,10,11],"import":[1,2,4,8,9,10],"int":[1,3,4,7,9,10,11],"long":[1,4,6,9],"new":[0,3,4,7,11],"public":[0,7],"return":[1,2,3,4,6,7,9,11],"true":[1,2,3,4,6,9,11],"try":[1,4,8,9,11],"void":[1,4,7,9,10],"while":[1,3,4,5,9],A:[1,3,7,8,9,11],And:[1,3,4,10,11],As:[1,4,8,9],At:[3,11],Be:4,But:[1,4],By:[3,6,9,11],For:[0,1,2,3,4,7,8,11],If:[0,1,2,3,4,6,7,8,9,11],In:[1,2,4,6,9,11,12],It:[1,3,4,6,7,8,9,10,11],Not:0,Of:9,On:[4,11],One:[3,4],Or:[7,8],That:[1,4,6,9],The:[0,1,2,3,4,6,8,9,10,11],Then:[0,4,7,10],There:[2,4,5,6,8,9,12],These:[4,8,9,10,11],To:[0,2,4,6,7,8,9,10,11],With:6,_:[2,4],__device__:10,__global:7,__global__:[1,4,7,9,10],__init__:3,__kernel:7,__shared__:[4,9],__syncthread:[4,9],_cost_func:3,_funcptr:3,ab:7,abl:[0,1,3,4],about:[0,1,3,4,7,9,11],abov:[1,3,4,8,9],abruptli:3,absolut:[2,11],acceler:7,accept:[2,3,11],acceptance_prob:3,access:[1,4],accord:11,account:[6,9],achiev:2,across:[6,9],actual:[0,1,2,3,4,8,9,10],ad:[4,6,11],add:[0,1,3,4,6,9],addit:[0,1,3,4,8],address_mod:11,addtion:4,adjac:11,adjust:1,advanc:[3,10,11],advis:3,affect:[4,9],after:[0,1,2,3,4,6,8,9,11],again:[1,4,9],against:[2,3],aggress:11,algebra:9,algorithm:[3,5,7,11],all:[0,1,3,4,5,6,7,8,9,11],allclos:[2,11],alloc:[1,3,4,5,6,11],allow:[1,2,3,4,7,9,10,11],almost:[2,4],along:[1,3,8,12],alpha:[3,11],alreadi:[1,3,4,8,9,11],also:[0,1,3,4,6,7,8,9,10,11,12],altern:11,although:2,alwai:[1,3,4],amd:8,among:[4,7],amount:[1,4,9],an:[0,1,2,3,4,5,6,7,8,9,10,11],analysi:[4,7],analyz:4,ani:[0,1,3,4,6,7,9,10,11,12],anneal:[3,7,11],anoth:[3,4,6,9,11],answer:[1,2,3,4,5,7,11],anyth:1,api:[1,3,7],app:8,append:[3,11],appl:8,appli:4,applic:[0,1,4,5,6,10,11],appropi:4,approx:4,approxim:4,ar:[0,1,2,3,4,5,6,7,8,9,10,11,12],arch:4,architectur:3,area:[4,9],arg:[2,3,4,6,7,9,10],argument:[1,2,3,4,5,6,7,9,10,11],arithmet:[4,11],around:[1,5],arrai:[1,2,3,4,11],articl:7,artifact:3,arxiv:7,assum:[1,3,4,9,11],assumpt:4,astyp:[1,2,4,6,7,9,10],atol:[2,3,11],attempt:[3,10],author:7,auto:[7,9,10,11,12],automat:[1,4,9,10,11],auxilliari:11,avail:[1,4,5,7,8],averag:[3,4,6],avoid:[1,9,12],ax1:4,ax2:4,axesimag:4,axi:11,b0:11,b:[7,9,10,11],back:[6,11],backend:[3,6],bandwidth:9,base:[3,7,10,11],bash:8,basic:[1,3,4],basin:11,basinhop:[7,11],bayes_opt:11,bayesian:[7,11],becaus:[1,2,4,6,8,9,10,12],becom:4,been:[1,4,6,9],befor:[0,1,2,3,4,6,8,9,11],begin:[1,4],behavior:[1,7,9,11],behaviour:3,behind:6,beignet:8,being:[3,4,9,11],below:[0,5,6,8,9],ben:7,benchmark:[1,2,3,4,5,6,7,9,11,12],benchmarkobserv:11,benefit:9,benvanwerkhoven:8,best1bin:11,best1exp:11,best2bin:11,best2exp:11,best:[3,4,9,10,11,12],best_tim:4,beta:3,better:[0,4],between:[3,4,6,7,9,11],beyond:[4,11],bfg:[7,11],bind:8,biologi:4,bit:[1,3,4,6,9],block:[1,3,4,5,9,11,12],block_size_:12,block_size_i:[1,2,4,6,9,11],block_size_nam:[1,3,11],block_size_str:4,block_size_x:[1,2,3,4,6,7,9,10,11],block_size_z:[1,4,11],blockdim:[1,11],blockidx:[1,4,7,9,10],boilerpl:4,bool:[3,11],border:[6,11],border_s:1,both:[4,5,7,9],bottom:3,bound:[1,3,9],boundari:4,bracket:3,branch:0,briefli:9,brute:7,brute_forc:11,buffer:3,build:[3,4,8],built:11,bulk:4,bx:4,c1:11,c2:11,c:[0,1,5,6,7,8,9,10,11],c_arg:3,cach:[3,4,8,9,11],cachefil:[3,11],calcul:3,call:[1,2,3,4,6,9,10,11],callabl:[2,3,11],can:[0,1,2,3,4,6,7,8,9,10,11,12],cannot:[0,4],capabl:[0,3,4,9,11],care:4,cartesian:1,caus:4,cc:4,cd:8,cedric:7,cell:[1,4,9],center:5,central:4,certain:[1,3,4,12],cg:[7,11],chanc:[3,8,10],chang:[0,7,11],changelog:0,check:[0,2,3,4,6,9],check_argument_list:3,check_argument_typ:3,check_kernel_output:3,check_restrict:3,check_tune_params_list:3,chemistri:4,children:3,choic:[6,8],choos:[4,9,11],chosen:11,chunk:6,circumst:1,cite:7,clamp:11,clarifi:6,clean:[5,9],cleaner:4,cleanup:4,cleanup_lib:3,clock:12,clone:[1,4,8,9],close:[3,4],closer:4,closest:3,cltune:7,cmem_arg:[2,3,11],cobyla:[7,11],code:[1,3,8,9,10,11,12],cognit:11,collabor:9,collect:[1,4,9],color:4,column:9,com:[7,8],combin:[1,3,4,5,7,9,11],come:[3,4,9,10],command:[0,7,8],common:10,commonli:[1,4,8,9],commun:4,compact:3,compar:[1,2,4,9],comparison:2,compat:[0,8],compil:[0,1,2,3,4,5,6,7,8,9,10,11,12],compile_and_benchmark:3,compile_kernel:3,compiler_opt:[3,11],compiler_opt_:12,complain:3,complet:1,complex:[6,9],compos:[1,3,9],comprehens:7,comput:[1,2,3,5,6,7,9,11],compute_capability_major:4,compute_capability_minor:4,compute_intens:3,concentr:4,concept:4,condens:4,condit:[4,9],config:3,config_valid:3,configur:[1,3,4,5,9,11],confus:1,consid:[0,9,11],consist:[5,9,11],consol:[7,11],constant:[1,3,4,5,6,9,11],constantrbf:11,construct:[2,9],consumpt:9,contain:[1,3,4,6,7,9,10,11],contant:9,content:[0,1,3,7,11],context:[3,4],continu:[1,3,4,8,11],continuum:8,contrast:1,control:[4,7,11],conv_filt:1,conveni:[4,6,11],convent:[3,6,11],convert:4,convolut:[2,6,9],convolution_correct:2,convolution_kernel:[1,2],convolution_na:[1,2],convolution_stream:[6,7],cooler:4,copi:[0,3,4,11],copy_constant_memory_arg:3,copy_shared_memory_arg:3,copy_texture_memory_arg:3,core_freq:12,correct:[6,11],correctli:9,correspond:[1,4],correspondingli:1,cost:[3,4],could:[1,2,3,4,6,8,9,10,11],coupl:9,cours:[1,4,8,9],covariancekernel:11,covariancelengthscal:11,cover:[4,11],cpu:[2,6],cpu_result:2,creat:[0,1,3,4,9,11],create_device_target:11,create_kernel_inst:3,creation:[1,3],criterion:11,crossov:[3,11],csv:[4,5],ctype:3,cu:[1,2,6,9,10],cub:5,cuda:[0,1,2,4,5,6,7,10,11],cudamemcpytosymbol:6,cudastreamwaitev:6,cupi:[10,11],current:[1,2,3,4,7,9,11],current_problem_s:3,custom:[2,5],d:4,d_filter:2,data:[1,3,4,6,9,11],datafram:4,debug:3,decreas:[1,9],deep:1,def:[2,4],defin:[1,2,3,4,5,7,9,10,11],definit:1,degrad:4,degre:4,delet:3,delete_temp_fil:3,delta:4,demonstr:[2,5,9],demot:10,denot:[9,11],depend:[1,2,5,7,11],deriv:[1,3,4],descret:4,describ:[0,1,3,6,11],design:[0,4,7],dest:3,detail:[3,7,8,11],detect:[3,7,10,11],detect_languag:3,determin:[1,3,4],dev:[0,8],develop:[3,7,8],devic:[1,2,4,5,6,7,10,11],device_nam:[3,11],device_opt:3,devicealloc:3,devprop:4,df:4,dict:[1,2,3,6,7,10,11],dictionari:[1,3,4,9,11],did:[1,4,9],diff_evo:11,differ:[1,2,3,4,5,6,9,11],differenti:[3,7,11],difficult:[4,10],diffuse_kernel:4,dim:3,dimens:[1,3,4,5,6,9,11,12],dimension:[5,11],dir:8,direct:[3,4,6,9,11],directli:[3,4,6,9,10,11],directori:[0,1,4,7,8,9],discard:3,discontinu:9,discuss:[0,3],displai:1,disrupt:3,disruptive_uniform:11,disruptive_uniform_crossov:3,distanc:[3,4],distance_to:3,distant:4,distinct:9,distribut:9,divid:[1,4,6,9,11],divison:11,divisor:[1,3,4,9,11],dna1:3,dna2:3,dna:3,doc:[0,8],docstr:0,document:[1,2,4,8,9,12],doe:[2,3,4,6,7,9,10,11],doi:7,domain:[1,4,5,11],don:[3,4,6,11],done:[1,7,8],doubl:[4,10],doubt:0,down:9,download:8,dramat:9,drastic:9,driver:[3,4],drv:4,dt:4,dtarget_gpu:11,dtype:3,dual:11,due:[9,10,11],dump:[3,4],dump_cach:3,dure:[3,4,7,11],dynam:11,e:[0,8,11],each:[1,2,3,4,9,11],earlier:[3,4],easi:[4,7,11],easiest:7,easili:4,effect:[1,4,11],effici:9,ei:11,either:[3,10,11],element:[2,3,4,9,11],ellipsi:1,empti:11,enabl:[7,10],end:[1,3,4,9,11],energi:12,enough:[1,2,9],ensur:[0,2,4,6,8],enter:[1,4,9],entir:[3,4,9,11],entri:[0,3,4],env:[1,11],environ:[1,3,8,11],ep:3,equal:[4,9,11],equat:[1,3,4],equi:4,error:[0,1,2,6,9,10],essenti:1,estim:4,euclidian:3,evalu:[3,4,9,11],even:[0,4,6,9],event:[4,6],everi:[1,2,4,5,7],everyth:[1,3,4,8],everywher:4,evolut:[3,7,11],exact:7,exactli:[1,3,4,7,9],exampl:[0,2,3,4,6,8,9,11],exce:11,except:[3,5],exchang:4,execut:[1,3,4,5,6,7,9,11],exist:[3,11],expand:[1,9],expect:[0,1,2,3,4,7,9,11],experi:1,explain:[1,3,4,6,8,9,10,11],explan:7,expos:3,express:[3,4,5,6,9,11],extens:[3,7],extern:10,extra:10,f:[1,2,6],f_h:1,f_w:1,fact:[4,6],factor:[1,4,5,9,12],fail:[1,3,8,11],fals:[3,11],familiar:[1,9],far:[1,4,9],fast:[2,4,11],faster:[4,9],fastest:3,featur:[1,2,5,8,10,11],feel:4,few:[1,4,6,10],fewer:[1,4],field:[2,4],field_copi:4,fifth:9,fig:4,figur:9,file:[0,1,3,4,5,6,9,10,11],filenam:[1,3,5,9,11],fill:[3,9],filter:[1,2,5,6],filter_height:1,filter_heigth:1,filter_mod:11,filter_s:1,filter_width:1,find:[1,3,6,7,9],fine:4,finish:[1,6],firefli:[3,7,11],firefly_algorithm:11,first:[0,1,2,4,6,7,8,9,10,11],first_kernel:2,fit:[3,6],five:1,fix:[4,11],flag:0,flexibl:[2,3,4,9],float32:[1,2,3,4,6,7,9,10,11],flori:7,fly:4,follow:[0,1,2,3,4,6,7,8,9,10,11],forbidden:3,forc:[7,10],foreseen:3,fork:0,form:[3,9],format:[3,4],formula:4,fortran:[5,10],fortun:9,found:[1,3,7,11],four:[3,4],fourth:9,fp:4,frac:4,fraction:[3,11],free:[1,4,6,8,9],freeli:1,frequent:9,from:[1,2,3,5,6,7,8,9,10,11],full:[7,8],fulli:8,func:[3,11],further:[4,7,8,9],futur:[3,7,11,12],g:8,gamma:11,gcc:3,geforc:4,gene:3,gener:[0,1,3,4,7,9,11,12],genet:[3,7,11],genetic_algorithm:[7,11],get:[4,7,8,9],get_attribut:4,get_bound:3,get_bounds_x0_ep:3,get_config_str:3,get_devic:4,get_environ:3,get_funct:4,get_global_id:7,get_grid_dimens:3,get_initial_condit:4,get_instance_str:3,get_kernel_str:[3,4],get_local_s:11,get_number_of_valid_config:3,get_problem_s:3,get_smem_arg:3,get_temp_filenam:3,get_thread_block_dimens:3,gflop:[1,3,5,9],gh:0,giga:[1,9],git:8,github:[0,1,4,8,9],give:4,given:[3,4,11],global:[3,4],go:[1,4,9],goe:9,good:[2,4,12],googl:0,got:4,gpu:[0,1,2,3,5,6,7,9,11,12],gpu_arg:3,gpu_result:[2,4],grain:4,graphic:12,great:[3,4],greedi:11,greedy:11,greedy_il:11,greedy_ml:11,grid:[1,3,4,5,6,9,11,12],grid_div:3,grid_div_i:[1,2,4,6,9,11],grid_div_x:[1,2,4,6,9,11],grid_div_z:11,grid_size_:12,grid_size_i:6,grid_size_x:6,group:[3,4,11],grow:4,gt:4,gtx:4,guarante:3,guess:[3,4],guid:7,h:[1,11],ha:[1,3,4,6,9,11],had:1,half:4,halt:6,ham:11,hand:9,handl:[6,11],happen:[0,1,9],hardwar:[3,4],have:[0,1,3,4,6,7,8,9,10,11,12],haven:[1,8],header:11,header_filenam:11,heat:4,height:9,help:[0,10],helper:3,here:[1,6,7,9,11],high:[3,4,7,9],highest:3,highli:[1,9],hold:[0,4,9,11],home:8,hop:11,host:[0,3,5,10,11],hot:4,hotspot:4,how:[0,1,2,3,4,5,7,9,10,11],howev:[1,2,4,6,8,9,10,11],html:0,http:[7,8],i:[1,2,4,6,7,8,9,10,11],i_like_convolut:1,id:3,idea:[4,6,9,12],ignor:[1,3,4,11],illeg:1,illustr:5,imag:[1,4],image_height:1,image_width:1,impact:[4,6],implement:[2,3,5,7],importantli:4,impos:9,improv:[0,4,9,11],imshow:4,includ:[0,1,2,3,4,6,9,10,11],increas:[1,4],independ:9,index:[3,7],indic:[1,3,12],individu:3,inertia:11,influenc:1,info:11,inform:[1,3,4,7,11,12],init:4,initi:[3,4],inlin:4,inner:9,input:[1,2,4,5,6,7,9,11],input_imag:1,input_s:[1,2,6],input_width:1,insert:[1,2,3,6,7,9,10,12],insid:[4,6,7,9,10,11],inspect:3,instal:[0,1,4,6,9],instanc:[2,3,4,6,11],instant:4,instanti:[3,10],instead:[1,5,9,11],instruct:[4,5,9],int32:[3,7,10,11],integ:[3,11],integr:[0,10],intel:8,intend:7,intens:3,interact:3,interest:1,interfac:[1,2,6,11],intermedi:4,intern:[3,7,10],interpret:1,introduc:[4,9],io:8,isclos:2,isol:10,item:[3,4],iter:[3,4,9,11],iterfac:3,its:[2,3,4,7,8,9,11],itself:[6,11],j:[1,4,7,9],jan:7,join:4,journal:7,json:[3,4,5,11],jupyt:[1,4,8,9],just:[1,2,3,4,6,7,8,9],k:[4,9],kb:9,keep:[4,9],kei:[3,4,9,11],kepler:11,kernel:[0,1,3,6,11,12],kernel_argu:3,kernel_inst:3,kernel_nam:[1,3,6,10,11],kernel_opt:3,kernel_sourc:[1,3,11],kernel_str:[1,2,3,4,6,7,11],kernel_tun:[1,2,4,6,7,8,9,10,11,12],kernelinst:3,kernelsourc:3,kerneltun:7,keyword:3,know:[1,4,9],known:9,l1:4,l2:4,l:[7,11],lambda:[1,3,4,9,11],lang:[3,5,6,10,11],languag:[3,6,7,9,11],larg:[3,4,11],larger:[3,4,6,10],last:3,later:[4,11],latest:8,latter:6,launch:[3,4,6,11],lcb:11,learn:1,left:[3,4],let:[1,4,7,10],level:[0,3],lh:11,librari:[3,5,7],like:[1,3,4,5,7,9,10,11],likewis:4,limit:[1,4,5,9,10,11,12],limits_:1,line:[1,4],linear:[1,9,11],link:0,linkag:10,linux:8,list:[0,1,2,3,4,5,6,7,8,9,11],littl:[1,3,4,9],ll:[1,4,8,9],load:3,local:[0,11],locat:[0,2],lock:5,log:11,longer:[1,3],look:[0,1,3,4,8,9,10,11],looks_like_a_filenam:3,lookup:3,loop:[4,5,9,12],loop_unroll_factor_:12,loss:4,lot:[1,4,9,11],low:[3,4,9],lower:3,lowest:3,lt:4,made:3,mai:[1,2,3,4,6,7,8,9,11],main:[1,3],maintain:3,make:[0,1,4,7,8,9,10],make_context:4,manag:[4,9],mani:[1,4,7,9,11],manual:8,map:[2,5],match:[0,1,2,3,7],matern32:11,matern52:11,matlab:10,matmul:9,matmul_kernel:9,matmul_na:9,matmul_shar:9,matplotlib:[4,8],matric:9,matrix:7,matter:[4,6],max:3,max_fev:11,max_thread:3,maxim:11,maximum:[2,11],maxit:11,md:0,mead:[7,11],mean:[0,1,6,7,9,10,12],meant:1,measur:[3,4,6,9,11,12],meet:3,melt:4,mem:3,mem_alloc:4,mem_freq:12,member:3,memcpi:[3,6],memcpy_dtoh:[3,4],memcpy_htod:[3,4],memori:[1,3,5,6,11,12],memset:3,merg:[4,9],messi:4,metal:4,method:[3,4,6,7,9,11],metric:[1,3,5,9,11],millisecond:4,mimick:1,min:4,mind:4,miniconda3:8,miniconda:8,minim:[0,10,11],mirror:11,miss:3,ml:11,mode:0,model:[4,7],modif:3,modul:[0,3,6,7],moment:[4,11],more:[2,3,4,7,8,9,10,11],most:[3,4,5,6,7,9,11],mostli:[3,11],motion:4,move:[1,3,4,6,9,11],move_toward:3,ms:4,much:[1,4,10,11],multi:11,multipl:[3,6,7,10,11],multiprocessor:4,must:[3,11],mutat:[3,11],mutation_ch:[3,11],my_typ:10,n:[2,3,4,6,7,9,10],naiv:[1,2,4],name:[1,2,3,4,9,11,12],name_of_gpu:11,namelijk:9,nativ:8,nbyte:4,ndarrai:3,ndrang:3,nearest:[3,11],necessari:[2,3,4,11],necessarili:[2,6],need:[1,2,3,4,6,7,8,9,10,11],neighbor:[3,11],nelder:[7,11],net:8,network:1,neural:1,new_cost:3,newer:[0,8],newli:9,next:[4,9],nice:4,nieuwpoort:7,no_improv:11,non:2,none:[2,3,7,11],nonumb:1,normal:[3,7,11],normalize_verify_funct:3,normalized_coordin:11,notat:11,note:[0,1,3,4,7,8,9,11],notebook:[1,4,9],notic:[1,4],now:[1,3,4,6,9],np:[1,9],nugteren:7,num_stream:6,number:[1,2,3,4,5,7,9,11,12],numer:4,numpi:[1,2,3,4,6,7,8,9,10,11],nvcc:3,nvidia:[8,9,10],nvml:12,nvml_:12,nvml_energi:12,nvml_gr_clock:12,nvml_mem_clock:12,nvml_power:12,nvml_pwr_limit:12,nvmlobserv:12,nvrtc:[3,10],nx:4,ny:4,o:1,obj:3,object:[1,2,3,4,11],objective_higher_is_bett:11,observ:[3,11,12],obtain:[1,4],occup:9,occur:11,occurr:3,offer:3,often:4,old:1,old_cost:3,older:8,omit:3,omp_get_wtim:6,onc:[2,3,4,11],one:[0,1,3,4,8,9,11],ones:[4,12],onli:[1,2,3,4,5,6,7,8,9,11],open:[2,4,6,9],open_cach:3,opencl:[0,1,4,5,6,7,9,11],openmp:6,oper:[1,4,6,9],opportun:9,optim:[1,2,3,4,6,7,9,11],option:[1,2,3,4,5,6,7,8,9,10,11,12],order:[1,2,3,4,6,7,9,11],ordered_greedy_ml:11,ordereddict:[1,3,4,9,11],org:7,os:1,other:[1,3,4,6,9,11,12],otherwis:[9,11],our:[1,4,9],out:[1,2,9],outer:9,output:[0,1,2,3,4,5,6,7,9,11,12],output_imag:1,output_s:1,over:[3,4,8,9],overhead:[4,9],overlap:[5,6,7],overview:7,overwritten:11,own:[1,3,6,8],p:[1,3,9,11],packag:0,page:[0,1,4,5,7,9],pair:4,panda:[4,5,8],parallel:[1,4],param:[1,2,3,11],paramet:[2,3,4,5,6,7,9,10,11],parameter_spac:3,parametr:1,parent:3,pars:4,part:[4,9,11],partial:[2,4,5],particl:[3,7,11],particular:[1,4,5,6,9],particularli:1,pass:[0,2,3,4,5,6,7,9,10,11],path:1,per:[0,1,4,11],percentag:11,perform:[1,2,3,4,5,6,7,9,11],permut:11,physic:7,pick:9,pii:7,pip:[0,1,4,7,8,9],pipelin:5,pixel:1,place:[1,3,4,11],plai:4,plain:6,platform:[3,7,8,11],pleas:[0,1,7,8],plot:4,plu:11,pmb:7,po:3,poi:11,point:[1,3,4,6,9,11],pop:4,pop_siz:3,popsiz:11,popul:[3,11],posit:[2,3,10,11],possibl:[0,1,2,4,6,7,9,11],powel:[7,11],power:[3,9,12],power_read:12,powerful:7,powersensor:12,pragma:[4,9],precis:6,precomput:2,prefer:[1,3,4,11],prefix:8,prepar:[3,4],prepare_kernel_str:3,prepend:3,preprocessor:[1,3,7],present:[0,9],press:[1,4,9],pretti:9,previou:[4,11],previous:[3,4,9],print:[0,1,3,4,7,9,11],print_config_output:3,probabl:[3,11],problem:[0,1,3,4,5,6,9,11],problem_s:[1,2,3,4,6,9,11,12],problemat:1,proce:9,process:[1,3,4,7,9,10],process_cach:3,process_metr:3,prod:[1,2,6],produc:[0,2],product:[1,4,11],profil:9,program:[2,4,6,7,9,10],programm:9,promis:1,properti:[3,9,11],propos:0,provid:[2,3,4,6,7,10,11],ps_energi:12,ps_power:12,pso:11,pull:0,purpos:[4,6,9,11,12],put:[0,4],py:[2,6],pycuda:[0,3,4,6,10],pylint:0,pyopencl:[0,3],pyplot:4,pytest:0,python:[0,1,3,5,6,7,9,10,11],pythonpath:8,qualiti:[3,6],quantiti:4,queue:7,quick:4,quickli:4,quiet:[3,11],quit:[4,9,10],r:[2,6],race:9,radiat:4,rais:3,rand1bin:11,rand1exp:11,rand2bin:11,rand2exp:11,randint:4,randn:[1,2,6,7,9,10],random:[1,2,3,4,6,7,9,10,11],random_popul:3,random_sampl:11,random_v:3,randomli:3,randomwalk:11,randtobest1bin:11,randtobest1exp:11,rang:[1,2,4,6,10],rather:[4,11],rawkernel:3,rbf:11,re:[1,4,9],read:[1,2,3,4,6,9,11],read_cach:3,read_fil:3,readi:[1,3,4,9],ready_argument_list:3,real:10,realiti:9,realiz:9,realli:[1,4,8],reason:[1,3,11],recent:[3,8],recommend:8,record:[1,4,11],redistribut:4,reduc:[4,9],reduct:[2,11],redund:9,ref:11,refer:[1,2,3,4,5,6,11],reflect:2,regard:[0,3],regardless:10,region:4,regist:[1,4,9],regular:3,reject:11,rel:3,relat:12,releas:3,relev:[3,7],rememb:[1,4,9],remov:0,repeatedli:3,replac:[1,2,3,4,7,9,11],replace_param_occurr:3,repo:8,report:[11,12],repositori:[0,1,4,8,9],repres:[3,4],represent:3,reproduc:0,request:[0,11],requir:[0,1,3,4,6,7,8,9,10],research:7,reserv:12,resolut:3,resourc:11,respect:9,respons:3,rest:[3,4],restart:[4,11],restrict:[3,5,9,10,11],result:[0,1,2,3,7,9,11,12],result_host:3,results_filenam:11,retri:8,retriev:[1,3,11],reus:[1,4,9],rewrit:10,right:[1,4,7,8],risk:10,roadmap:0,rob:7,robust:3,room:9,roughli:9,round:[4,11],row:9,run:[1,2,3,4,6,8,9,11],run_kernel:[1,2,3,5,11],runtim:[1,3,4,7,10],s0167739x18313359:7,s:[1,3,4,5,6,8,9,10,11],sa:9,safer:11,sai:[3,4,10],same:[0,1,2,3,4,6,7,11],sampl:[3,11],sample_fract:3,samplingmethod:11,satisfi:11,save:4,sb:9,sc21:7,scalar:[1,4,11],scale:3,scienc:7,sciencedirect:7,scipi:5,script:[1,3,7,9,10],sdk:8,search:[1,3,5,9,11],second:[1,2,4,9,11],secondli:[1,9],section:[3,4],see:[0,1,3,4,6,7,8,9,10,11],seem:4,seen:[0,1,3,9],select:[0,1,3,4,7,9,11],self:3,semant:2,sensibl:9,separ:[5,6,10],seper:10,seri:3,serv:4,session:3,set:[1,2,3,4,5,9,10,11,12],set_titl:4,setup:[1,4,6],setup_block_and_grid:3,setup_method_argu:3,setup_method_opt:3,sever:[3,4,5,8,9,10,11],sh:8,sh_u:4,share:[1,3,11],sheet:4,shift:[1,4,9],shortli:1,should:[0,1,2,3,4,6,9,11],show:[1,4,5,7],shown:[1,3],shuffl:5,signal:[1,12],signatur:[1,3],signific:0,significantli:9,silent:1,similar:[3,6,9,11],similarli:1,simpl:[1,3,4,5,6,7,9],simpli:[1,2,3,4,11],simplic:4,simplifi:[4,7],simul:[3,7,11],simulated_ann:11,simulation_mod:11,sinc:[1,9,10],singl:[1,2,3,4,6,9,10,11],single_point:11,single_point_crossov:3,size:[1,2,3,4,5,6,7,9,10,11],skip:[0,1,4,11],skippablefailur:3,slightli:[6,9,10],slowest:3,slsqp:[7,11],sm_:4,small:[1,4,9],smem_arg:[3,11],snap:3,snap_to_nearest_config:3,snippet:2,so:[1,3,4,6,8,9,10,11],social:11,softwar:[1,3,4,7],solut:[7,9],some:[1,3,4,8,9,10,11],somehow:4,someth:[1,4,9],sometim:4,somewher:1,soon:11,sort:11,sourc:[0,1,3,6,10,11],sourcemodul:4,space:[1,2,3,6,7,9,11],spatial:4,special:[1,4,12],specif:[1,3,4,11],specifi:[1,2,3,4,6,7,9,10,11,12],speed:[3,7],spent:6,sphinxdoc:0,split:6,spread:6,squar:9,src:3,sriniva:11,stai:1,stand:9,start:[2,4,6,7,8,9,11],state:[3,4,11],statement:[1,9,10],stdout:4,step:[4,9,10],still:[0,2,7,9],store:[1,3,9,11],store_cach:3,store_result:11,str:[3,4],strategi:[1,11],strategy_opt:[3,11],stream:[3,4,7],string:[1,3,4,5,9,11],structur:[1,4,9],style:0,submatric:9,subplot:4,sudo:8,suffix:[3,11],suit:11,sum:[1,2,9],sum_float:2,sum_x:2,summar:11,supercomput:7,suppli:[3,6,9,10,11],support:[1,3,4,6,7,8,10,11,12],suppos:4,sure:[1,4,7,8,9],swarm:[3,7,11],symbol:[3,11],symlink:0,synchron:[4,9],system:[7,8],t:[1,3,4,6,8,10,11],t_min:11,take:[1,3,4,9,10,11],target:11,techniqu:9,tell:[1,4,5,6,9],temp_x:3,temperatur:[4,11,12],templat:7,temporari:3,term:1,terminolog:4,test:[4,5,7,9,11],test_vector_add:5,test_vector_add_parameter:5,texmem_arg:[3,11],text:[4,9],textur:[3,11],than:[1,3,4,11,12],thank:0,thei:[3,4,5,9],them:[0,1,6,9],therefor:[1,2,3,4,6,9],thi:[0,1,2,3,4,6,7,8,9,10,11,12],thing:[1,6,7,9],think:4,third:[2,9],those:[1,5],thousand:4,thread:[1,3,4,5,11,12],threadidx:[1,4,7,9,10],three:[1,2,3,9],through:[3,4,7,11],ti:4,tiker:8,tile:[1,5,9],tile_size_i:[1,2,4,6,9,11],tile_size_x:[1,2,4,6,9],time:[1,3,4,6,7,9,10,11,12],time_sinc:4,titan:4,titl:7,tj:4,tnc:[7,11],to_csv:4,togeth:[4,8,11],token:1,toler:2,too:[1,4,6,9,11],took:[1,3,4,11],toolkit:[7,8],top:[0,3,11],total:[1,4,9],toward:3,track:0,transfer:[5,6,7],treat:11,tri:4,troubl:8,trust:2,trusti:1,tunabl:[3,4,5,7,9,10,11,12],tune:[2,3,5,8,10,11,12],tune_kernel:[1,2,3,4,6,7,9,10,11],tune_param:[1,2,3,4,6,7,9,10,11],tune_params_kei:3,tuner:[0,1,2,3,6,9,10,11,12],tuning_opt:3,tupl:[3,11],turn:3,tutori:[0,1,7],two:[1,3,4,5,9,11],two_point:11,two_point_crossov:3,tx:[4,9],ty:[4,9],type:[0,1,2,3,4,5,7,8,9,10,11],typenam:10,typic:[3,8,9,11],typicali:11,u:4,u_:4,u_new:4,u_old:4,undefin:[1,3,4,9],under:[1,11],understand:1,underutil:4,uniform:[3,11],uniform_crossov:3,uniformli:3,uniqu:[3,11],unit:[0,3,7],unless:1,unload:3,unrol:[4,5,9,12],unscale_and_snap_to_nearest:3,unsign:3,until:6,up:[1,3,4,7,9,11],updat:[0,3],upload:6,url:7,us:[0,1,2,3,5,6,7,8,10,11,12],usag:9,usecas:5,user:[1,2,3,5,7,8,9,10,11],util:9,v:[0,3,4],valid:[3,5,9],valu:[1,2,3,4,5,6,9,11],van:7,vari:[4,9],variabl:[3,8,11],ve:[1,4,8,9],vector:6,vector_add:[7,10],verbos:[1,2,3,4,6,11],veri:[2,4,6,8,9,10],verif:[5,11],verifi:[2,3,5,7,11],verify_partial_reduc:2,version:[0,1,8,9,11],virtual:8,visual:9,vocabulari:7,volum:7,w:[1,4,11],wa:[1,3,4,11],wai:[1,4,6,7,8,9,11],want:[2,6,8,9,11,12],warp:11,warpsiz:9,we:[1,2,3,4,5,6,8,9,10],weight:[1,3],weighted_choic:3,well:[4,7,9,11],went:4,were:[1,4,9,11],werkhoven:7,wget:8,what:[0,1,2,3,4,6,7,9,10,11,12],whatev:[3,6],when:[0,1,3,4,6,8,9,10,11,12],whenev:2,where:[0,1,2,3,4,9,10,11],whether:[3,11],which:[1,3,4,5,6,7,9,10,11,12],whole:[3,4,9],whose:[2,11],why:[4,6],wide:[4,8,9],width:9,wiki:8,willemsen2021bayesian:7,willemsen:7,wish:3,within:[4,9,11],without:[4,6,11],won:1,word:9,work:[0,1,3,4,8,10,11],workshop:7,worri:4,worst:4,would:[1,4,7,10],wrap:[3,10,11],wrapper:10,write:[1,5,9,10,11],write_fil:3,writefil:[1,9],written:[0,10],wrote:1,www:7,x0:3,x1:3,x2:3,x86_64:8,x:[1,2,3,4,7,9,10,11],x_:4,x_i:4,xn:3,xyz:11,y1:3,y2:3,y:[1,3,4,6,9,11],y_:4,year:7,yet:[1,6],yield:9,yn:3,you:[0,1,2,3,4,6,7,8,9,10,11,12],your:[0,1,4,6,7,8,11],yourself:[6,11],z:[1,3,11],zero:[1,2,6,9],zeros_lik:[4,7,9,10]},titles:["Contribution guide","Getting Started","Kernel Correctness Verification","Design documentation","Tutorial: From physics to tuned GPU kernels","Kernel Tuner Examples","Tuning Host Code","The Kernel Tuner documentation","Installation Guide","Matrix multiplication tutorial","Templated kernels","API Documentation","Parameter Vocabulary"],titleterms:{"2d":1,"function":3,The:7,add:5,api:11,auto:4,backend:10,basinhop:3,brute_forc:3,build:0,c:3,cfunction:3,citat:7,code:[0,4,5,6,7],comput:4,contribut:[0,7],convolut:[1,5],convolution_correct:5,convolution_stream:5,core:3,correct:[2,7],cuda:[3,8,9],cudafunct:3,cupi:3,cupyfunct:3,depend:8,design:3,develop:0,devic:3,deviceinterfac:3,diff_evo:3,diffus:4,document:[0,3,7,11],exampl:[1,5,7,10],expdist:5,firefly_algorithm:3,from:4,gener:5,genetic_algorithm:3,get:1,gpu:4,guid:[0,8],host:[6,7],implement:[1,4],increas:9,indic:7,instal:[7,8],interfac:3,introduct:7,issu:0,kernel:[2,4,5,7,8,9,10],kernel_tun:3,matrix:[5,9],memori:[4,9],minim:3,more:1,multipl:[5,9],naiv:9,number:6,opencl:[3,8],openclfunct:3,packag:8,paramet:[1,12],per:9,physic:4,point:5,polygon:5,pso:3,py:5,pycuda:8,pyopencl:8,python:[4,8],random_sampl:3,reduct:5,relat:7,report:0,result:4,run:0,runner:3,search:7,select:10,sepconv:5,sequenti:3,sequentialrunn:3,setup:0,share:[4,9],simulated_ann:3,simulationrunn:3,spars:5,start:1,stencil:5,store:4,strategi:[3,7],stream:6,tabl:7,templat:10,test:[0,1],thread:9,tile:4,tunabl:1,tune:[1,4,6,7,9],tuner:[4,5,7,8],tutori:[4,8,9],us:[4,9],usag:7,util:3,vector:5,verif:[2,7],vocabulari:12,work:[7,9]}})