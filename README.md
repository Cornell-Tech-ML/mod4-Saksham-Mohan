# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py



* Sentiment Analysis Logs

Epoch    Train Loss    Train Accuracy    Validation Accuracy
50       13.5990       0.8511            0.7300
49       14.1956       0.8578            0.6900
48       15.1399       0.8222            0.6900
47       15.2583       0.8422            0.6100
46       15.5970       0.8356            0.6000
45       14.1397       0.8533            0.6900
44       15.7266       0.8400            0.7200
43       15.5444       0.8422            0.6900
42       16.2342       0.8289            0.7000
41       15.3607       0.8422            0.6700
40       16.0886       0.8556            0.7200
39       16.5856       0.8511            0.6900
38       16.8054       0.8578            0.6900
37       17.4826       0.8289            0.7100
36       18.6080       0.8222            0.7400
35       18.8006       0.8044            0.6700
34       18.5591       0.8244            0.7100
33       19.2258       0.8244            0.7500
32       19.4843       0.8067            0.6700
31       20.2259       0.7978            0.7100
30       20.8394       0.7911            0.7500
29       21.1886       0.7978            0.6600
28       21.4017       0.7844            0.6700
27       21.9492       0.7622            0.7200
26       22.1362       0.7956            0.6900
25       22.4455       0.7800            0.7100
24       23.2465       0.7800            0.6800
23       24.0037       0.7822            0.6900
22       24.7114       0.7511            0.6100
21       24.8828       0.7533            0.7000
20       25.7429       0.7400            0.7000
19       26.0507       0.7467            0.7000
18       26.5511       0.7356            0.7200
17       26.8753       0.7333            0.6800
16       27.5529       0.7022            0.6300
15       27.4375       0.7178            0.6500
14       28.4979       0.6867            0.5900
13       28.5209       0.6911            0.7000
12       29.0548       0.6800            0.5900
11       29.4338       0.6733            0.5800
10       29.6585       0.6556            0.5600
9        29.8152       0.6533            0.5400
8        30.2058       0.6044            0.6800
7        30.3096       0.5844            0.6300
6        30.5563       0.5867            0.5500
5        30.5903       0.5733            0.6000
4        30.7992       0.5800            0.5500
3        30.9974       0.5667            0.5600
2        31.1580       0.5111            0.5000
1        31.3450       0.5222            0.5000


* MNIST Logs

Epoch: 1/25, loss: 2.308614881071967, correct: 1
Epoch: 1/25, loss: 11.530926023262552, correct: 1
Epoch: 1/25, loss: 11.50485018417473, correct: 1
Epoch: 1/25, loss: 11.516206307677779, correct: 1
Epoch: 1/25, loss: 11.488259675112966, correct: 1
Epoch: 1/25, loss: 11.500431362579667, correct: 1
Epoch: 1/25, loss: 11.51550942806234, correct: 1
Epoch: 1/25, loss: 11.51935804969738, correct: 1
Epoch: 1/25, loss: 11.504077563031007, correct: 1
Epoch: 1/25, loss: 11.434990272554595, correct: 2
Epoch: 1/25, loss: 11.436445750874793, correct: 2
Epoch: 1/25, loss: 11.432877574144436, correct: 2
Epoch: 1/25, loss: 11.348450465053627, correct: 2
Epoch: 2/25, loss: 2.2979070839109874, correct: 2
Epoch: 2/25, loss: 11.391710795055264, correct: 2
Epoch: 2/25, loss: 11.372065036349696, correct: 2
Epoch: 2/25, loss: 11.361634389028902, correct: 4
Epoch: 2/25, loss: 11.279436780941158, correct: 3
Epoch: 2/25, loss: 11.299724414838895, correct: 2
Epoch: 2/25, loss: 11.126389027374392, correct: 6
Epoch: 2/25, loss: 11.27918996323765, correct: 8
Epoch: 2/25, loss: 11.213643177768912, correct: 8
Epoch: 2/25, loss: 10.7917177269489, correct: 6
Epoch: 2/25, loss: 10.81079094309624, correct: 8
Epoch: 2/25, loss: 10.734680047972438, correct: 8
Epoch: 2/25, loss: 10.422855762278925, correct: 8
Epoch: 3/25, loss: 2.079846178663194, correct: 10
Epoch: 3/25, loss: 10.104412252994225, correct: 7
Epoch: 3/25, loss: 10.296151257354413, correct: 8
Epoch: 3/25, loss: 10.05417738278266, correct: 8
Epoch: 3/25, loss: 9.467321851343101, correct: 9
Epoch: 3/25, loss: 8.83595026241867, correct: 8
Epoch: 3/25, loss: 7.873350327647566, correct: 11
Epoch: 3/25, loss: 8.483322455033125, correct: 13
Epoch: 3/25, loss: 8.42015407327735, correct: 12
Epoch: 3/25, loss: 6.694856416625636, correct: 13
Epoch: 3/25, loss: 7.5957955967781565, correct: 13
Epoch: 3/25, loss: 6.997936226384415, correct: 13
Epoch: 3/25, loss: 7.182737407934246, correct: 13
Epoch: 4/25, loss: 1.2551186257020495, correct: 12
Epoch: 4/25, loss: 5.826882976399958, correct: 12
Epoch: 4/25, loss: 6.420039561497383, correct: 11
Epoch: 4/25, loss: 5.398532552079915, correct: 13
Epoch: 4/25, loss: 4.456294004824018, correct: 13
Epoch: 4/25, loss: 4.961112041685381, correct: 13
Epoch: 4/25, loss: 4.392457870500115, correct: 12
Epoch: 4/25, loss: 5.230617939571161, correct: 14
Epoch: 4/25, loss: 6.527547228695696, correct: 13
Epoch: 4/25, loss: 4.2414166612835, correct: 13
Epoch: 4/25, loss: 4.901165105706109, correct: 14
Epoch: 4/25, loss: 4.987880204401128, correct: 13
Epoch: 4/25, loss: 4.5211927259118365, correct: 14
Epoch: 5/25, loss: 0.6946392718165302, correct: 12
Epoch: 5/25, loss: 4.348819917509893, correct: 13
Epoch: 5/25, loss: 4.733342378567431, correct: 15
Epoch: 5/25, loss: 3.9029746435864547, correct: 15
Epoch: 5/25, loss: 3.602378343618855, correct: 14
Epoch: 5/25, loss: 4.084842176334457, correct: 13
Epoch: 5/25, loss: 4.192986112762151, correct: 12
Epoch: 5/25, loss: 4.437414004500306, correct: 12
Epoch: 5/25, loss: 5.468223364463907, correct: 13
Epoch: 5/25, loss: 3.023243660834748, correct: 13
Epoch: 5/25, loss: 4.194071042192907, correct: 14
Epoch: 5/25, loss: 4.85811467082167, correct: 14
Epoch: 5/25, loss: 5.497935542692176, correct: 13
Epoch: 6/25, loss: 0.8683325492075347, correct: 12
Epoch: 6/25, loss: 4.326909636152971, correct: 13
Epoch: 6/25, loss: 5.401032490370614, correct: 13
Epoch: 6/25, loss: 4.029339326418453, correct: 13
Epoch: 6/25, loss: 3.9961018817563874, correct: 14
Epoch: 6/25, loss: 3.8977685670210467, correct: 14
Epoch: 6/25, loss: 3.953372462343544, correct: 12
Epoch: 6/25, loss: 5.044222941340096, correct: 13
Epoch: 6/25, loss: 4.4158014581289615, correct: 14
Epoch: 6/25, loss: 2.924255330196079, correct: 13
Epoch: 6/25, loss: 3.563880960625137, correct: 15
Epoch: 6/25, loss: 4.242306263062116, correct: 14
Epoch: 6/25, loss: 6.255708208679907, correct: 12
Epoch: 7/25, loss: 1.1805896467102857, correct: 12
Epoch: 7/25, loss: 5.292825820634033, correct: 12
Epoch: 7/25, loss: 5.98102945959976, correct: 14
Epoch: 7/25, loss: 4.173975504914734, correct: 13
Epoch: 7/25, loss: 3.618403095926198, correct: 13
Epoch: 7/25, loss: 3.250120756236895, correct: 14
Epoch: 7/25, loss: 4.742153037505637, correct: 12
Epoch: 7/25, loss: 5.324216382609395, correct: 13
Epoch: 7/25, loss: 4.504856089747607, correct: 15
Epoch: 7/25, loss: 2.9892307296338183, correct: 16
Epoch: 7/25, loss: 4.403835466212616, correct: 15
Epoch: 7/25, loss: 4.210905447583791, correct: 13
Epoch: 7/25, loss: 5.067807562848198, correct: 14
Epoch: 8/25, loss: 0.7346841311427315, correct: 15
Epoch: 8/25, loss: 3.964024975495682, correct: 13
Epoch: 8/25, loss: 4.328870606074001, correct: 14
Epoch: 8/25, loss: 3.577102839324632, correct: 15
Epoch: 8/25, loss: 3.594930669860063, correct: 15
Epoch: 8/25, loss: 3.419185109621142, correct: 14
Epoch: 8/25, loss: 3.9821580237488594, correct: 15
Epoch: 8/25, loss: 5.6060734555565945, correct: 15
Epoch: 8/25, loss: 4.278994322499647, correct: 15
Epoch: 8/25, loss: 3.821142877732064, correct: 15
Epoch: 8/25, loss: 3.5792076612946544, correct: 15
Epoch: 8/25, loss: 4.426929063810663, correct: 14
Epoch: 8/25, loss: 4.7561443512947505, correct: 15
Epoch: 9/25, loss: 0.5295982268176727, correct: 15
Epoch: 9/25, loss: 3.7704474790205627, correct: 14
Epoch: 9/25, loss: 3.9942819159100567, correct: 14
Epoch: 9/25, loss: 3.1508943654177175, correct: 15
Epoch: 9/25, loss: 3.2132168696144627, correct: 15
Epoch: 9/25, loss: 2.5396899458660167, correct: 14
Epoch: 9/25, loss: 3.436292655602457, correct: 14
Epoch: 9/25, loss: 4.709874340314414, correct: 14
Epoch: 9/25, loss: 4.144594638958289, correct: 14
Epoch: 9/25, loss: 3.3268666911002134, correct: 14
Epoch: 9/25, loss: 3.092354282072077, correct: 15
Epoch: 9/25, loss: 4.067157482800097, correct: 15
Epoch: 9/25, loss: 3.3741818730226347, correct: 15
Epoch: 10/25, loss: 0.40707277674276293, correct: 15
Epoch: 10/25, loss: 3.162477594825987, correct: 14
Epoch: 10/25, loss: 3.083096386264962, correct: 15
Epoch: 10/25, loss: 3.0426085597951023, correct: 14
Epoch: 10/25, loss: 2.3981560365694374, correct: 14
Epoch: 10/25, loss: 2.2021391120292093, correct: 15
Epoch: 10/25, loss: 2.526640310835524, correct: 15
Epoch: 10/25, loss: 3.5225536905848007, correct: 15
Epoch: 10/25, loss: 3.009394159350048, correct: 14
Epoch: 10/25, loss: 2.218580712243121, correct: 14
Epoch: 10/25, loss: 2.5393971980650423, correct: 15
Epoch: 10/25, loss: 2.909910155762261, correct: 15
Epoch: 10/25, loss: 2.550411908988433, correct: 15
Epoch: 11/25, loss: 0.3204298330839849, correct: 15
Epoch: 11/25, loss: 2.4585876064117285, correct: 15
Epoch: 11/25, loss: 2.2469084930750696, correct: 14
Epoch: 11/25, loss: 2.4170772545881096, correct: 14
Epoch: 11/25, loss: 2.628023919416629, correct: 14
Epoch: 11/25, loss: 1.9174844424968993, correct: 15
Epoch: 11/25, loss: 2.3402678679422158, correct: 15
Epoch: 11/25, loss: 2.8579960368472253, correct: 15
Epoch: 11/25, loss: 2.3807890105716245, correct: 14
Epoch: 11/25, loss: 2.174577653175071, correct: 15
Epoch: 11/25, loss: 1.7701849771444336, correct: 14
Epoch: 11/25, loss: 3.689393674876989, correct: 14
Epoch: 11/25, loss: 2.6767654563149623, correct: 15
Epoch: 12/25, loss: 0.26092356784999904, correct: 15
Epoch: 12/25, loss: 2.3600595173276164, correct: 15
Epoch: 12/25, loss: 2.0254677920303403, correct: 15
Epoch: 12/25, loss: 1.8209751089296682, correct: 14
Epoch: 12/25, loss: 2.037222608128656, correct: 14
Epoch: 12/25, loss: 1.914718693755081, correct: 16
Epoch: 12/25, loss: 2.0099590999501458, correct: 15
Epoch: 12/25, loss: 2.080277327228233, correct: 16
Epoch: 12/25, loss: 1.9787268945211522, correct: 15
Epoch: 12/25, loss: 1.9116239264041592, correct: 14
Epoch: 12/25, loss: 1.6073972584189002, correct: 16
Epoch: 12/25, loss: 2.624451119748239, correct: 14
Epoch: 12/25, loss: 2.164948306085004, correct: 15
Epoch: 13/25, loss: 0.3034853727810975, correct: 15
Epoch: 13/25, loss: 1.6774702887072162, correct: 15
Epoch: 13/25, loss: 1.9164439040750434, correct: 15
Epoch: 13/25, loss: 2.410338866984218, correct: 14
Epoch: 13/25, loss: 1.9524086882554479, correct: 14
Epoch: 13/25, loss: 1.6091849802936937, correct: 15
Epoch: 13/25, loss: 1.5909064975545242, correct: 15
Epoch: 13/25, loss: 2.0890669131425206, correct: 15
Epoch: 13/25, loss: 2.1418980008773336, correct: 14
Epoch: 13/25, loss: 1.4352471554625115, correct: 14
Epoch: 13/25, loss: 1.1131693037633539, correct: 16
Epoch: 13/25, loss: 2.377635492659518, correct: 14
Epoch: 13/25, loss: 1.678272185379769, correct: 14
Epoch: 14/25, loss: 0.2055537373241525, correct: 15
Epoch: 14/25, loss: 1.620774600720471, correct: 15
Epoch: 14/25, loss: 1.4806726264879722, correct: 15
Epoch: 14/25, loss: 1.7769594404844746, correct: 14
Epoch: 14/25, loss: 1.3472224902773364, correct: 14
Epoch: 14/25, loss: 1.135043558303862, correct: 15
Epoch: 14/25, loss: 1.5568354465931953, correct: 15
Epoch: 14/25, loss: 1.7955330356334365, correct: 15
Epoch: 14/25, loss: 1.661378370054912, correct: 15
Epoch: 14/25, loss: 1.24869432461524, correct: 14
Epoch: 14/25, loss: 1.53373666271747, correct: 15
Epoch: 14/25, loss: 1.7961182744531665, correct: 15
Epoch: 14/25, loss: 1.0896194734565214, correct: 15
Epoch: 15/25, loss: 0.3716649788491694, correct: 15
Epoch: 15/25, loss: 1.2871577987270724, correct: 14
Epoch: 15/25, loss: 2.0065774630480226, correct: 14
Epoch: 15/25, loss: 1.537207012400618, correct: 14
Epoch: 15/25, loss: 1.5849677917667875, correct: 14
Epoch: 15/25, loss: 1.2400437642574405, correct: 15
Epoch: 15/25, loss: 1.5761612332448371, correct: 14
Epoch: 15/25, loss: 1.82835895950617, correct: 16
Epoch: 15/25, loss: 1.6136919614853156, correct: 16
Epoch: 15/25, loss: 1.3435760878996446, correct: 15
Epoch: 15/25, loss: 1.0002253205306166, correct: 16
Epoch: 15/25, loss: 1.7447669078828187, correct: 15
Epoch: 15/25, loss: 1.2525831448511582, correct: 15
Epoch: 16/25, loss: 0.13911938176721392, correct: 15
Epoch: 16/25, loss: 1.4685505345183125, correct: 15
Epoch: 16/25, loss: 1.424679726851097, correct: 15
Epoch: 16/25, loss: 0.9371203670531272, correct: 14
Epoch: 16/25, loss: 1.4115407766353656, correct: 14
Epoch: 16/25, loss: 1.5827532747480162, correct: 16
Epoch: 16/25, loss: 1.24723501528656, correct: 15
Epoch: 16/25, loss: 1.780496727216015, correct: 16
Epoch: 16/25, loss: 1.7169488428029203, correct: 15
Epoch: 16/25, loss: 0.8926092931230895, correct: 15
Epoch: 16/25, loss: 1.1811592840293252, correct: 15
Epoch: 16/25, loss: 1.6996248012590485, correct: 15
Epoch: 16/25, loss: 1.5776124839362908, correct: 15
Epoch: 17/25, loss: 0.187382108027297, correct: 15
Epoch: 17/25, loss: 1.7136708921227712, correct: 15
Epoch: 17/25, loss: 1.4080567131875679, correct: 15
Epoch: 17/25, loss: 1.189640614504902, correct: 15
Epoch: 17/25, loss: 1.2013027376098528, correct: 15
Epoch: 17/25, loss: 1.5449805513110126, correct: 15
Epoch: 17/25, loss: 1.573634623649289, correct: 15
Epoch: 17/25, loss: 1.720461580704692, correct: 15
Epoch: 17/25, loss: 1.3788480273760984, correct: 15
Epoch: 17/25, loss: 1.7099376678370586, correct: 15
Epoch: 17/25, loss: 2.1855335406643173, correct: 15
Epoch: 17/25, loss: 2.1525367804430022, correct: 15
Epoch: 17/25, loss: 2.0027787471250704, correct: 15
Epoch: 18/25, loss: 0.23626034277760918, correct: 15
Epoch: 18/25, loss: 1.7654494948172181, correct: 15
Epoch: 18/25, loss: 1.9498062977838546, correct: 15
Epoch: 18/25, loss: 1.8852996498412025, correct: 15
Epoch: 18/25, loss: 1.1391851586988777, correct: 15
Epoch: 18/25, loss: 1.3989957948283278, correct: 15
Epoch: 18/25, loss: 1.4496093471391296, correct: 15
Epoch: 18/25, loss: 1.86831134031309, correct: 15
Epoch: 18/25, loss: 1.2046856752056914, correct: 15
Epoch: 18/25, loss: 0.999873789100062, correct: 15
Epoch: 18/25, loss: 1.4856971733113102, correct: 16
Epoch: 18/25, loss: 1.8448403231931199, correct: 16
Epoch: 18/25, loss: 1.2837883726699988, correct: 15
Epoch: 19/25, loss: 0.19405698672227206, correct: 15
Epoch: 19/25, loss: 1.3373260799736801, correct: 15
Epoch: 19/25, loss: 1.1414024555738622, correct: 15
Epoch: 19/25, loss: 0.9958416024497461, correct: 15
Epoch: 19/25, loss: 1.2022557056950751, correct: 14
Epoch: 19/25, loss: 1.730669721593618, correct: 15
Epoch: 19/25, loss: 1.259734289044049, correct: 15
Epoch: 19/25, loss: 1.083954951934286, correct: 15
Epoch: 19/25, loss: 1.1894854878055856, correct: 15
Epoch: 19/25, loss: 1.0030106228686044, correct: 15
Epoch: 19/25, loss: 1.2196313582769494, correct: 15
Epoch: 19/25, loss: 1.3186456595690186, correct: 15
Epoch: 19/25, loss: 1.138161630941259, correct: 15
Epoch: 20/25, loss: 0.05830363855542686, correct: 15
Epoch: 20/25, loss: 1.2436711874399333, correct: 15
Epoch: 20/25, loss: 1.4048768221695438, correct: 15
Epoch: 20/25, loss: 0.6413563567701195, correct: 15
Epoch: 20/25, loss: 0.9315393438580568, correct: 15
Epoch: 20/25, loss: 0.8177518927344178, correct: 15
Epoch: 20/25, loss: 0.8616184432557519, correct: 15
Epoch: 20/25, loss: 1.368487617252491, correct: 15
Epoch: 20/25, loss: 0.9687226532279194, correct: 15
Epoch: 20/25, loss: 0.37219744799397475, correct: 15
Epoch: 20/25, loss: 0.6927554215901075, correct: 15
Epoch: 20/25, loss: 1.104224837211453, correct: 14
Epoch: 20/25, loss: 0.8997513344843289, correct: 14
Epoch: 21/25, loss: 0.08537956076105135, correct: 14
Epoch: 21/25, loss: 0.7641705451499082, correct: 14
Epoch: 21/25, loss: 1.2397984544947733, correct: 15
Epoch: 21/25, loss: 1.0751513616594874, correct: 14
Epoch: 21/25, loss: 0.9228972260304138, correct: 14
Epoch: 21/25, loss: 0.9056426270099682, correct: 16
Epoch: 21/25, loss: 0.7161456133648709, correct: 16
Epoch: 21/25, loss: 1.237652176524736, correct: 15
Epoch: 21/25, loss: 1.196835303392005, correct: 16
Epoch: 21/25, loss: 0.6103184456486948, correct: 16
Epoch: 21/25, loss: 0.6008766302756703, correct: 16
Epoch: 21/25, loss: 1.0261386753594255, correct: 16
Epoch: 21/25, loss: 0.9057382171043825, correct: 16
Epoch: 22/25, loss: 0.09769477649882796, correct: 15
Epoch: 22/25, loss: 1.0059798972283445, correct: 14
Epoch: 22/25, loss: 1.191750079746114, correct: 14
Epoch: 22/25, loss: 0.9268853796792007, correct: 15
Epoch: 22/25, loss: 0.791634329780116, correct: 14
Epoch: 22/25, loss: 0.4973353221996181, correct: 16
Epoch: 22/25, loss: 1.3724170490118697, correct: 16
Epoch: 22/25, loss: 0.6749822363711273, correct: 16
Epoch: 22/25, loss: 0.6854381538821579, correct: 16
Epoch: 22/25, loss: 0.5787924361158447, correct: 16
Epoch: 22/25, loss: 0.6139706772176795, correct: 16
Epoch: 22/25, loss: 1.228229156284981, correct: 15
Epoch: 22/25, loss: 0.8124356954336395, correct: 15
Epoch: 23/25, loss: 0.022495730747411236, correct: 15
Epoch: 23/25, loss: 0.615185985123542, correct: 15
Epoch: 23/25, loss: 0.9949787773886367, correct: 16
Epoch: 23/25, loss: 0.9121776850405199, correct: 16
Epoch: 23/25, loss: 0.3716348646252086, correct: 16
Epoch: 23/25, loss: 0.4942868853389002, correct: 16
Epoch: 23/25, loss: 0.5898867506519594, correct: 16
Epoch: 23/25, loss: 0.9596802544630466, correct: 16
Epoch: 23/25, loss: 0.8044562691765169, correct: 16
Epoch: 23/25, loss: 0.4326702487137125, correct: 16
Epoch: 23/25, loss: 0.6261758260523456, correct: 16
Epoch: 23/25, loss: 1.012672648952251, correct: 15
Epoch: 23/25, loss: 0.6308045514405906, correct: 16
Epoch: 24/25, loss: 0.09686146492876624, correct: 16
Epoch: 24/25, loss: 0.7670947868818018, correct: 16
Epoch: 24/25, loss: 0.9817280052045505, correct: 16
Epoch: 24/25, loss: 0.698723751937752, correct: 16
Epoch: 24/25, loss: 0.9660879854448314, correct: 16
Epoch: 24/25, loss: 0.4084613876199178, correct: 16
Epoch: 24/25, loss: 0.49821040352378915, correct: 16
Epoch: 24/25, loss: 0.480262435413218, correct: 16
Epoch: 24/25, loss: 0.5107072348506332, correct: 16
Epoch: 24/25, loss: 0.38320239479070634, correct: 16
Epoch: 24/25, loss: 0.5296986909315439, correct: 16
Epoch: 24/25, loss: 0.5729297046164453, correct: 15
Epoch: 24/25, loss: 0.6473710825925043, correct: 15
Epoch: 25/25, loss: 0.06437476683415522, correct: 15
Epoch: 25/25, loss: 0.7242225885785807, correct: 15
Epoch: 25/25, loss: 1.1526489319543378, correct: 15
Epoch: 25/25, loss: 0.9746441913152704, correct: 16
Epoch: 25/25, loss: 0.4210142075181837, correct: 16
Epoch: 25/25, loss: 0.6070110771534634, correct: 16
Epoch: 25/25, loss: 0.8903376250591224, correct: 16
Epoch: 25/25, loss: 0.6162965957753549, correct: 16
Epoch: 25/25, loss: 0.5020907921513478, correct: 16
Epoch: 25/25, loss: 0.3412891566055577, correct: 16
Epoch: 25/25, loss: 0.7257221762857271, correct: 16
Epoch: 25/25, loss: 0.8054282703950533, correct: 16
Epoch: 25/25, loss: 0.44107805294417224, correct: 16