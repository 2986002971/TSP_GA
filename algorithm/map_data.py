map_data = [
    [116.46, 39.92],
    [117.2, 39.13],
    [121.48, 31.22],
    [106.54, 29.59],
    [91.11, 29.97],
    [87.68, 43.77],
    [106.27, 38.47],
    [111.65, 40.82],
    [108.33, 22.84],
    [126.63, 45.75],
    [125.35, 43.88],
    [123.38, 41.8],
    [114.48, 38.03],
    [112.53, 37.87],
    [101.74, 36.56],
    [117, 36.65],
    [113.65, 34.76],
    [118.78, 32.04],
    [117.27, 31.86],
    [120.19, 30.26],
    [119.3, 26.08],
    [115.89, 28.68],
    [113, 28.21],
    [114.31, 30.52],
    [113.23, 23.16],
    [121.5, 25.05],
    [110.35, 20.02],
    [103.73, 36.03],
    [108.95, 34.27],
    [104.06, 30.67],
    [106.71, 26.57],
    [102.73, 25.04],
    [114.1, 22.2],
    [113.33, 22.13],
]

simplified_data = [
    [116.46, 39.92],
    [117.2, 39.13],
    [121.48, 31.22],
    [106.54, 29.59],
    [91.11, 29.97],
    [87.68, 43.77],
    [106.27, 38.47],
    [111.65, 40.82],
    [108.33, 22.84],
    [126.63, 45.75],
]

distance_matrix = [
    [
        0.0,
        1.0824509226750245,
        10.044421337239898,
        14.321846947932372,
        27.232792732292438,
        29.0363720185563,
        10.292647861459168,
        4.893475247715052,
        18.916217909508234,
        11.722533855783912,
        9.732096382588903,
        7.170829798565854,
        2.737243138634192,
        4.432538775916118,
        15.09860920747338,
        3.314287253694228,
        5.875516998528722,
        8.2144263342001,
        8.100598743302871,
        10.355119506794695,
        14.128382780771481,
        11.254443566876153,
        12.21047501123523,
        9.642743385572386,
        17.068406486839947,
        15.700907617077432,
        20.816870562118602,
        13.311085605614583,
        9.398010427744792,
        15.47005171290645,
        16.531333884475263,
        20.246661453187777,
        17.876464974932826,
        18.063249984429714,
    ],
    [
        1.0824509226750245,
        0.0,
        8.993692234004902,
        14.305495447554412,
        27.651287492628626,
        29.88243631299161,
        10.949908675418264,
        5.801603226695184,
        18.548342243985044,
        11.521688244350297,
        9.433186100146644,
        6.732109624775868,
        2.9340074982862596,
        4.836992867474587,
        15.672156839439818,
        2.488051446413442,
        5.630222020489069,
        7.263917675744959,
        7.27033699356502,
        9.360395290798353,
        13.217885610036125,
        10.531789971320169,
        11.699846152834663,
        9.08208125927092,
        16.456056635780033,
        14.7219699768747,
        20.30060590228775,
        13.82211633578592,
        9.5750770231889,
        15.627898131226733,
        16.364403441616815,
        20.196757165446144,
        17.211475822834025,
        17.43493332364653,
    ],
    [
        10.044421337239898,
        8.993692234004902,
        0.0,
        15.028655961196263,
        30.39571351358609,
        36.05471536429042,
        16.84952818330532,
        13.740047307051022,
        15.593168375926687,
        15.415686815708211,
        13.238296718233807,
        10.74925113670715,
        9.766068809915279,
        11.150112107059734,
        20.449528111914965,
        7.0395525425981464,
        8.593049516906088,
        2.8217724926010628,
        4.2583682320814,
        1.608011193990891,
        5.583189052862175,
        6.140008143317078,
        8.998360961864114,
        7.204089116605931,
        11.53369411767106,
        6.17003241482571,
        15.789772005953735,
        18.390176725632628,
        12.895867555151149,
        17.42868038607628,
        15.48468275425752,
        19.742211122364182,
        11.65438973091256,
        12.208628096555325,
    ],
    [
        14.321846947932372,
        14.305495447554412,
        15.028655961196263,
        0.0,
        15.434678487095226,
        23.596016612979405,
        8.884103781473964,
        12.337949586539896,
        6.983308671396387,
        25.782817922019298,
        23.622451185260168,
        20.800713930055373,
        11.587803933446578,
        10.219515644099769,
        8.462913210000451,
        12.619635493943548,
        8.790961267119767,
        12.482792155603645,
        10.967488317750778,
        13.666433331341421,
        13.233960102705455,
        9.394179048751408,
        6.6057550666066875,
        7.825458453023691,
        9.279062452640352,
        15.633719966789727,
        10.30053396674172,
        7.026357520081086,
        5.26407636722721,
        2.7049584100314785,
        3.024780983806926,
        5.934526097339199,
        10.571929814371632,
        10.087403035469531,
    ],
    [
        27.232792732292438,
        27.651287492628626,
        30.39571351358609,
        15.434678487095226,
        0.0,
        14.219876933363384,
        17.38032220644945,
        23.229595347315033,
        18.637738596728948,
        38.86745167875043,
        36.957620323824955,
        34.370071283021794,
        24.72085152255076,
        22.83038326441324,
        12.50699804109683,
        26.737885107091024,
        23.04334394136407,
        27.747320591365213,
        26.228185221246243,
        29.081445975054265,
        28.457129159491824,
        24.813554763475548,
        21.960639790315764,
        23.20651848080621,
        23.14455659545026,
        30.78568660920201,
        21.660565551250038,
        13.999571422011465,
        18.35090188519355,
        12.968905119554236,
        15.966214329013619,
        12.622571053474015,
        24.2675297465564,
        23.562555039723513,
    ],
    [
        29.0363720185563,
        29.88243631299161,
        36.05471536429042,
        23.596016612979405,
        14.219876933363384,
        0.0,
        19.33075528788256,
        24.150846776044933,
        29.402166586835055,
        39.00029358863852,
        37.67016060491379,
        35.754313026542675,
        27.40780180897403,
        25.54080069222576,
        15.800876557963475,
        30.172119580831566,
        27.488561257366673,
        33.23857548090772,
        31.896962237805646,
        35.20540015395365,
        36.23203692866301,
        31.992377217080943,
        29.71895018334261,
        29.744233054493098,
        32.82643142347337,
        38.65528165723282,
        32.83277935234846,
        17.818813091785884,
        23.29512609968016,
        20.974136454214268,
        25.651138376298224,
        24.027388538915336,
        34.106909857094934,
        33.55908371812317,
    ],
    [
        10.292647861459168,
        10.949908675418264,
        16.84952818330532,
        8.884103781473964,
        17.38032220644945,
        19.33075528788256,
        0.0,
        5.870851726964334,
        15.765167300095486,
        21.6223957969509,
        19.832158228493437,
        17.431035540093422,
        8.22178204527462,
        6.288688257498544,
        4.916197717748951,
        10.883257784321756,
        8.260054479239235,
        14.065738515982732,
        12.833241991016923,
        16.16077040242822,
        17.98035038590739,
        13.725469026594322,
        12.270309694543167,
        11.306816528094902,
        16.817779282652037,
        20.298997512192567,
        18.895737614605046,
        3.5221016453248404,
        4.98220834570374,
        8.107040150387805,
        11.90813167545606,
        13.888718443398583,
        18.056073770341104,
        17.799977528075704,
    ],
    [
        4.893475247715052,
        5.801603226695184,
        13.740047307051022,
        12.337949586539896,
        23.229595347315033,
        24.150846776044933,
        5.870851726964334,
        0.0,
        18.2839492451713,
        15.770393146652994,
        14.037578138696137,
        11.770866578124133,
        3.9740407647632385,
        3.078457405909656,
        10.786829932839407,
        6.783170350212352,
        6.381504524796643,
        11.310406712404287,
        10.576672444582933,
        13.581060341519724,
        16.606929276660388,
        12.859129052933561,
        12.68205819258057,
        10.637932129883138,
        17.7305386269002,
        18.593423568563157,
        20.840585404445818,
        9.25583599681844,
        7.084666541200085,
        12.674012782067091,
        15.081979313074267,
        18.126632340288698,
        18.78049253880206,
        18.76535371369269,
    ],
    [
        18.916217909508234,
        18.548342243985044,
        15.593168375926687,
        6.983308671396387,
        18.637738596728948,
        29.402166586835055,
        15.765167300095486,
        18.2839492451713,
        0.0,
        29.321631946397524,
        27.062187642539175,
        24.20710846011972,
        16.38775762574002,
        15.605796999833105,
        15.220594600737519,
        16.305980497964544,
        13.053306094625992,
        13.922733208677096,
        12.699763777330661,
        13.98985346599456,
        11.438465806217193,
        9.552968125143098,
        7.116586260279575,
        9.733591320781864,
        4.910437862349961,
        13.354137935486515,
        3.468832656672845,
        13.96911235547914,
        11.44680304713941,
        8.918620969634263,
        4.06660792307299,
        6.016643582596524,
        5.805385430787516,
        5.050158413356951,
    ],
    [
        11.722533855783912,
        11.521688244350297,
        15.415686815708211,
        25.782817922019298,
        38.86745167875043,
        39.00029358863852,
        21.6223957969509,
        15.770393146652994,
        29.321631946397524,
        0.0,
        2.2661200321253934,
        5.115173506343653,
        14.395169328632429,
        16.152535404697304,
        26.532399062278557,
        13.249411307677029,
        17.00766003893539,
        15.798310036203238,
        16.749379092969388,
        16.775389712313686,
        20.991374419032216,
        20.167610170766388,
        22.21325055006583,
        19.58916282029428,
        26.26534027953949,
        21.326202193545853,
        30.447845572388207,
        24.877467716791426,
        21.080151802109956,
        27.144268271589116,
        27.6528262570031,
        31.624580629630483,
        26.675895486374962,
        27.1070913231206,
    ],
    [
        9.732096382588903,
        9.433186100146644,
        13.238296718233807,
        23.622451185260168,
        36.957620323824955,
        37.67016060491379,
        19.832158228493437,
        14.037578138696137,
        27.062187642539175,
        2.2661200321253934,
        0.0,
        2.8648385643871834,
        12.344205118192091,
        14.15883116644873,
        24.71870749048178,
        11.04515278300848,
        14.834567738899567,
        13.54069791406632,
        14.483328346757869,
        14.564683312726027,
        18.80006648924413,
        17.90339632583717,
        19.95172674231481,
        17.331220384035277,
        24.00443292394136,
        19.219557747253187,
        28.183321308887642,
        23.00101954262027,
        19.008211383504754,
        25.05530283193559,
        25.43787923550232,
        29.43824043654783,
        24.425087512637493,
        24.850410459386786,
    ],
    [
        7.170829798565854,
        6.732109624775868,
        10.74925113670715,
        20.800713930055373,
        34.370071283021794,
        35.754313026542675,
        17.431035540093422,
        11.770866578124133,
        24.20710846011972,
        5.115173506343653,
        2.8648385643871834,
        0.0,
        9.665552234611317,
        11.539818022828603,
        22.265381200419633,
        8.199201180602897,
        12.0097668586863,
        10.789698790976507,
        11.667720428601294,
        11.972789983959457,
        16.24083741683291,
        15.1074319459,
        17.100657882081606,
        14.474228822289628,
        21.224328022342654,
        16.85517427972787,
        25.380096532519335,
        20.479633785788252,
        16.27654140166146,
        22.29662082020501,
        22.57967670273425,
        26.595490219208212,
        21.685903255340783,
        22.08871657656913,
    ],
    [
        2.737243138634192,
        2.9340074982862596,
        9.766068809915279,
        11.587803933446578,
        24.72085152255076,
        27.40780180897403,
        8.22178204527462,
        3.9740407647632385,
        16.38775762574002,
        14.395169328632429,
        12.344205118192091,
        9.665552234611317,
        0.0,
        1.9565530915362384,
        12.824527281736362,
        2.8731167745150885,
        3.373692339262728,
        7.373608343274005,
        6.771484327678827,
        9.642458192805396,
        12.885453038213285,
        9.455717846890314,
        9.930901268263622,
        7.511923854779148,
        14.922446180167649,
        14.756720502875968,
        18.47747277091757,
        10.93446386431452,
        6.687189245116366,
        12.757194048849458,
        13.845739416874787,
        17.51578145559027,
        15.834560303336499,
        15.941533803244909,
    ],
    [
        4.432538775916118,
        4.836992867474587,
        11.150112107059734,
        10.219515644099769,
        22.83038326441324,
        25.54080069222576,
        6.288688257498544,
        3.078457405909656,
        15.605796999833105,
        16.152535404697304,
        14.15883116644873,
        11.539818022828603,
        1.9565530915362384,
        0.0,
        10.869231803582078,
        4.633497599006607,
        3.3055256768024064,
        8.547011173503869,
        7.6542602516507054,
        10.797578432222657,
        13.595477189124328,
        9.784973173187547,
        9.671426988816073,
        7.562466528851547,
        14.726645918198752,
        15.646510793144902,
        17.982627727893384,
        8.990305890235323,
        5.07704638544892,
        11.116694652638433,
        12.710719885199266,
        16.144624492381354,
        15.748453892366701,
        15.76031725569,
    ],
    [
        15.09860920747338,
        15.672156839439818,
        20.449528111914965,
        8.462913210000451,
        12.50699804109683,
        15.800876557963475,
        4.916197717748951,
        10.786829932839407,
        15.220594600737519,
        26.532399062278557,
        24.71870749048178,
        22.265381200419633,
        12.824527281736362,
        10.869231803582078,
        0.0,
        15.260265397430025,
        12.04525217668772,
        17.629293803212885,
        16.225624795366127,
        19.495961120191026,
        20.449547672259165,
        16.196200171645206,
        14.018206019316457,
        13.945841674133556,
        17.651631652626346,
        22.867831117095477,
        18.646814741397527,
        2.059368835347383,
        7.56493225349706,
        6.330442322618541,
        11.158001613192214,
        11.562460810744403,
        18.94674642253915,
        18.508187377482436,
    ],
    [
        3.314287253694228,
        2.488051446413442,
        7.0395525425981464,
        12.619635493943548,
        26.737885107091024,
        30.172119580831566,
        10.883257784321756,
        6.783170350212352,
        16.305980497964544,
        13.249411307677029,
        11.04515278300848,
        8.199201180602897,
        2.8731167745150885,
        4.633497599006607,
        15.260265397430025,
        0.0,
        3.8463749167235326,
        4.941710230274535,
        4.7976035684495635,
        7.142002520301988,
        10.817342557208772,
        8.04692487848619,
        9.339892933005173,
        6.694251265078117,
        14.006891161139217,
        12.442266674525182,
        17.910315463441734,
        13.28447590234556,
        8.39445650414605,
        14.25496404765722,
        14.404530537299719,
        18.396331155966937,
        14.738130817712266,
        14.97662512050028,
    ],
    [
        5.875516998528722,
        5.630222020489069,
        8.593049516906088,
        8.790961267119767,
        23.04334394136407,
        27.488561257366673,
        8.260054479239235,
        6.381504524796643,
        13.053306094625992,
        17.00766003893539,
        14.834567738899567,
        12.0097668586863,
        3.373692339262728,
        3.3055256768024064,
        12.04525217668772,
        3.8463749167235326,
        0.0,
        5.806487750783597,
        4.638361779766637,
        7.9386144886875485,
        10.356876942399186,
        6.479506154021304,
        6.582172893505606,
        4.291060474987503,
        11.607600957992997,
        12.48625644458738,
        15.104886626519248,
        10.000964953443244,
        4.725473521246313,
        10.425746975636807,
        10.734975547247425,
        14.619329669995134,
        12.568058720422973,
        12.63405318969332,
    ],
    [
        8.2144263342001,
        7.263917675744959,
        2.8217724926010628,
        12.482792155603645,
        27.747320591365213,
        33.23857548090772,
        14.065738515982732,
        11.310406712404287,
        13.922733208677096,
        15.798310036203238,
        13.54069791406632,
        10.789698790976507,
        7.373608343274005,
        8.547011173503869,
        17.629293803212885,
        4.941710230274535,
        5.806487750783597,
        0.0,
        1.52069063257456,
        2.2707928130941366,
        5.982641557038162,
        4.431895756896816,
        6.933779633071706,
        4.721366327664057,
        10.471719056582828,
        7.500566645260875,
        14.681461098950612,
        15.569926139837657,
        10.079771822814244,
        14.783615931158383,
        13.251633861528177,
        17.51006853213316,
        10.896237882865814,
        11.309756849729354,
    ],
    [
        8.100598743302871,
        7.27033699356502,
        4.2583682320814,
        10.967488317750778,
        26.228185221246243,
        31.896962237805646,
        12.833241991016923,
        10.576672444582933,
        12.699763777330661,
        16.749379092969388,
        14.483328346757869,
        11.667720428601294,
        6.771484327678827,
        7.6542602516507054,
        16.225624795366127,
        4.7976035684495635,
        4.638361779766637,
        1.52069063257456,
        0.0,
        3.329624603465082,
        6.12611622481977,
        3.4665256381570275,
        5.617419336314493,
        3.2491845130740056,
        9.592267719366465,
        8.01679487076974,
        13.71393451931283,
        14.167586244664255,
        8.662014777175106,
        13.263491244766584,
        11.810914443852349,
        16.06001245329529,
        10.166833331967236,
        10.497452071812473,
    ],
    [
        10.355119506794695,
        9.360395290798353,
        1.608011193990891,
        13.666433331341421,
        29.081445975054265,
        35.20540015395365,
        16.16077040242822,
        13.581060341519724,
        13.98985346599456,
        16.775389712313686,
        14.564683312726027,
        11.972789983959457,
        9.642458192805396,
        10.797578432222657,
        19.495961120191026,
        7.142002520301988,
        7.9386144886875485,
        2.2707928130941366,
        3.329624603465082,
        0.0,
        4.273698632332424,
        4.581091572976901,
        7.476536631355454,
        5.885745492288971,
        9.9424141937459,
        5.372169021912845,
        14.201521045296525,
        17.442032565042407,
        11.933888720781667,
        16.135209945953598,
        13.975925729625216,
        18.22361105818492,
        10.102064145510068,
        10.637504406579582,
    ],
    [
        14.128382780771481,
        13.217885610036125,
        5.583189052862175,
        13.233960102705455,
        28.457129159491824,
        36.23203692866301,
        17.98035038590739,
        16.606929276660388,
        11.438465806217193,
        20.991374419032216,
        18.80006648924413,
        16.24083741683291,
        12.885453038213285,
        13.595477189124328,
        20.449547672259165,
        10.817342557208772,
        10.356876942399186,
        5.982641557038162,
        6.12611622481977,
        4.273698632332424,
        0.0,
        4.288134792657524,
        6.650330818839013,
        6.679348770651219,
        6.735822147295749,
        2.4291768153018434,
        10.808612306859748,
        18.4777541925419,
        13.198431725019454,
        15.91620871941556,
        12.599531737330564,
        16.602605217254304,
        6.488019728699969,
        7.158449552801219,
    ],
    [
        11.254443566876153,
        10.531789971320169,
        6.140008143317078,
        9.394179048751408,
        24.813554763475548,
        31.992377217080943,
        13.725469026594322,
        12.859129052933561,
        9.552968125143098,
        20.167610170766388,
        17.90339632583717,
        15.1074319459,
        9.455717846890314,
        9.784973173187547,
        16.196200171645206,
        8.04692487848619,
        6.479506154021304,
        4.431895756896816,
        3.4665256381570275,
        4.581091572976901,
        4.288134792657524,
        0.0,
        2.927968579066381,
        2.4252834885843746,
        6.1274790901316,
        6.681990721334473,
        10.280428006654201,
        14.208733229954033,
        8.911324256248339,
        11.996207734113309,
        9.419368344002697,
        13.654127581065,
        6.722685475314164,
        7.032503110557436,
    ],
    [
        12.21047501123523,
        11.699846152834663,
        8.998360961864114,
        6.6057550666066875,
        21.960639790315764,
        29.71895018334261,
        12.270309694543167,
        12.68205819258057,
        7.116586260279575,
        22.21325055006583,
        19.95172674231481,
        17.100657882081606,
        9.930901268263622,
        9.671426988816073,
        14.018206019316457,
        9.339892933005173,
        6.582172893505606,
        6.933779633071706,
        5.617419336314493,
        7.476536631355454,
        6.650330818839013,
        2.927968579066381,
        0.0,
        2.6555978611228017,
        5.0552349104665755,
        9.068384641158534,
        8.60805436785805,
        12.127872855534063,
        7.28876532754348,
        9.272281272696594,
        6.50028460915367,
        10.748106810038683,
        6.109836331686799,
        6.088949006191465,
    ],
    [
        9.642743385572386,
        9.08208125927092,
        7.204089116605931,
        7.825458453023691,
        23.20651848080621,
        29.744233054493098,
        11.306816528094902,
        10.637932129883138,
        9.733591320781864,
        19.58916282029428,
        17.331220384035277,
        14.474228822289628,
        7.511923854779148,
        7.562466528851547,
        13.945841674133556,
        6.694251265078117,
        4.291060474987503,
        4.721366327664057,
        3.2491845130740056,
        5.885745492288971,
        6.679348770651219,
        2.4252834885843746,
        2.6555978611228017,
        0.0,
        7.4388171102669265,
        9.034212749321325,
        11.221924968560433,
        11.928809664002523,
        6.541567090537253,
        10.25109750221897,
        8.565191182921728,
        12.811198226551644,
        8.32264981841721,
        8.447040901996392,
    ],
    [
        17.068406486839947,
        16.456056635780033,
        11.53369411767106,
        9.279062452640352,
        23.14455659545026,
        32.82643142347337,
        16.817779282652037,
        17.7305386269002,
        4.910437862349961,
        26.26534027953949,
        24.00443292394136,
        21.224328022342654,
        14.922446180167649,
        14.726645918198752,
        17.651631652626346,
        14.006891161139217,
        11.607600957992997,
        10.471719056582828,
        9.592267719366465,
        9.9424141937459,
        6.735822147295749,
        6.1274790901316,
        5.0552349104665755,
        7.4388171102669265,
        0.0,
        8.483218728760914,
        4.260751107492675,
        15.996465234544788,
        11.905901897798422,
        11.852805575052686,
        7.3578869249262135,
        10.66697707881666,
        1.295569372901345,
        1.0348429832588137,
    ],
    [
        15.700907617077432,
        14.7219699768747,
        6.17003241482571,
        15.633719966789727,
        30.78568660920201,
        38.65528165723282,
        20.298997512192567,
        18.593423568563157,
        13.354137935486515,
        21.326202193545853,
        19.219557747253187,
        16.85517427972787,
        14.756720502875968,
        15.646510793144902,
        22.867831117095477,
        12.442266674525182,
        12.48625644458738,
        7.500566645260875,
        8.01679487076974,
        5.372169021912845,
        2.4291768153018434,
        6.681990721334473,
        9.068384641158534,
        9.034212749321325,
        8.483218728760914,
        0.0,
        12.232064421020686,
        20.888592580640754,
        15.572761476372776,
        18.32315475020609,
        14.867901667686674,
        18.77000266382506,
        7.929848674470408,
        8.676133931654125,
    ],
    [
        20.816870562118602,
        20.30060590228775,
        15.789772005953735,
        10.30053396674172,
        21.660565551250038,
        32.83277935234846,
        18.895737614605046,
        20.840585404445818,
        3.468832656672845,
        30.447845572388207,
        28.183321308887642,
        25.380096532519335,
        18.47747277091757,
        17.982627727893384,
        18.646814741397527,
        17.910315463441734,
        15.104886626519248,
        14.681461098950612,
        13.71393451931283,
        14.201521045296525,
        10.808612306859748,
        10.280428006654201,
        8.60805436785805,
        11.221924968560433,
        4.260751107492675,
        12.232064421020686,
        0.0,
        17.324678929203852,
        14.318606775800504,
        12.368775202096607,
        7.493470491034179,
        9.124954794408564,
        4.337614551801485,
        3.6513696060519565,
    ],
    [
        13.311085605614583,
        13.82211633578592,
        18.390176725632628,
        7.026357520081086,
        13.999571422011465,
        17.818813091785884,
        3.5221016453248404,
        9.25583599681844,
        13.96911235547914,
        24.877467716791426,
        23.00101954262027,
        20.479633785788252,
        10.93446386431452,
        8.990305890235323,
        2.059368835347383,
        13.28447590234556,
        10.000964953443244,
        15.569926139837657,
        14.167586244664255,
        17.442032565042407,
        18.4777541925419,
        14.208733229954033,
        12.127872855534063,
        11.928809664002523,
        15.996465234544788,
        20.888592580640754,
        17.324678929203852,
        0.0,
        5.508720359575351,
        5.37014897372503,
        9.918265977478118,
        11.035402122260885,
        17.286000115700563,
        16.892897915988243,
    ],
    [
        9.398010427744792,
        9.5750770231889,
        12.895867555151149,
        5.26407636722721,
        18.35090188519355,
        23.29512609968016,
        4.98220834570374,
        7.084666541200085,
        11.44680304713941,
        21.080151802109956,
        19.008211383504754,
        16.27654140166146,
        6.687189245116366,
        5.07704638544892,
        7.56493225349706,
        8.39445650414605,
        4.725473521246313,
        10.079771822814244,
        8.662014777175106,
        11.933888720781667,
        13.198431725019454,
        8.911324256248339,
        7.28876532754348,
        6.541567090537253,
        11.905901897798422,
        15.572761476372776,
        14.318606775800504,
        5.508720359575351,
        0.0,
        6.072240113829493,
        8.019201955307029,
        11.13019766221607,
        13.12278171730369,
        12.90596761192279,
    ],
    [
        15.47005171290645,
        15.627898131226733,
        17.42868038607628,
        2.7049584100314785,
        12.968905119554236,
        20.974136454214268,
        8.107040150387805,
        12.674012782067091,
        8.918620969634263,
        27.144268271589116,
        25.05530283193559,
        22.29662082020501,
        12.757194048849458,
        11.116694652638433,
        6.330442322618541,
        14.25496404765722,
        10.425746975636807,
        14.783615931158383,
        13.263491244766584,
        16.135209945953598,
        15.91620871941556,
        11.996207734113309,
        9.272281272696594,
        10.25109750221897,
        11.852805575052686,
        18.32315475020609,
        12.368775202096607,
        5.37014897372503,
        6.072240113829493,
        0.0,
        4.88185415595345,
        5.784963266953389,
        13.135543384268496,
        12.604146143233978,
    ],
    [
        16.531333884475263,
        16.364403441616815,
        15.48468275425752,
        3.024780983806926,
        15.966214329013619,
        25.651138376298224,
        11.90813167545606,
        15.081979313074267,
        4.06660792307299,
        27.6528262570031,
        25.43787923550232,
        22.57967670273425,
        13.845739416874787,
        12.710719885199266,
        11.158001613192214,
        14.404530537299719,
        10.734975547247425,
        13.251633861528177,
        11.810914443852349,
        13.975925729625216,
        12.599531737330564,
        9.419368344002697,
        6.50028460915367,
        8.565191182921728,
        7.3578869249262135,
        14.867901667686674,
        7.493470491034179,
        9.918265977478118,
        8.019201955307029,
        4.88185415595345,
        0.0,
        4.263953564475101,
        8.585394574508502,
        7.97107270070974,
    ],
    [
        20.246661453187777,
        20.196757165446144,
        19.742211122364182,
        5.934526097339199,
        12.622571053474015,
        24.027388538915336,
        13.888718443398583,
        18.126632340288698,
        6.016643582596524,
        31.624580629630483,
        29.43824043654783,
        26.595490219208212,
        17.51578145559027,
        16.144624492381354,
        11.562460810744403,
        18.396331155966937,
        14.619329669995134,
        17.51006853213316,
        16.06001245329529,
        18.22361105818492,
        16.602605217254304,
        13.654127581065,
        10.748106810038683,
        12.811198226551644,
        10.66697707881666,
        18.77000266382506,
        9.124954794408564,
        11.035402122260885,
        11.13019766221607,
        5.784963266953389,
        4.263953564475101,
        0.0,
        11.719321652723751,
        10.992183586530926,
    ],
    [
        17.876464974932826,
        17.211475822834025,
        11.65438973091256,
        10.571929814371632,
        24.2675297465564,
        34.106909857094934,
        18.056073770341104,
        18.78049253880206,
        5.805385430787516,
        26.675895486374962,
        24.425087512637493,
        21.685903255340783,
        15.834560303336499,
        15.748453892366701,
        18.94674642253915,
        14.738130817712266,
        12.568058720422973,
        10.896237882865814,
        10.166833331967236,
        10.102064145510068,
        6.488019728699969,
        6.722685475314164,
        6.109836331686799,
        8.32264981841721,
        1.295569372901345,
        7.929848674470408,
        4.337614551801485,
        17.286000115700563,
        13.12278171730369,
        13.135543384268496,
        8.585394574508502,
        11.719321652723751,
        0.0,
        0.7731752712031043,
    ],
    [
        18.063249984429714,
        17.43493332364653,
        12.208628096555325,
        10.087403035469531,
        23.562555039723513,
        33.55908371812317,
        17.799977528075704,
        18.76535371369269,
        5.050158413356951,
        27.1070913231206,
        24.850410459386786,
        22.08871657656913,
        15.941533803244909,
        15.76031725569,
        18.508187377482436,
        14.97662512050028,
        12.63405318969332,
        11.309756849729354,
        10.497452071812473,
        10.637504406579582,
        7.158449552801219,
        7.032503110557436,
        6.088949006191465,
        8.447040901996392,
        1.0348429832588137,
        8.676133931654125,
        3.6513696060519565,
        16.892897915988243,
        12.90596761192279,
        12.604146143233978,
        7.97107270070974,
        10.992183586530926,
        0.7731752712031043,
        0.0,
    ],
]

simplified_matrix = [
    [
        0.0,
        1.0824509226750245,
        10.044421337239898,
        14.321846947932372,
        27.232792732292438,
        29.0363720185563,
        10.292647861459168,
        4.893475247715052,
        18.916217909508234,
        11.722533855783912,
    ],
    [
        1.0824509226750245,
        0.0,
        8.993692234004902,
        14.305495447554412,
        27.651287492628626,
        29.88243631299161,
        10.949908675418264,
        5.801603226695184,
        18.548342243985044,
        11.521688244350297,
    ],
    [
        10.044421337239898,
        8.993692234004902,
        0.0,
        15.028655961196263,
        30.39571351358609,
        36.05471536429042,
        16.84952818330532,
        13.740047307051022,
        15.593168375926687,
        15.415686815708211,
    ],
    [
        14.321846947932372,
        14.305495447554412,
        15.028655961196263,
        0.0,
        15.434678487095226,
        23.596016612979405,
        8.884103781473964,
        12.337949586539896,
        6.983308671396387,
        25.782817922019298,
    ],
    [
        27.232792732292438,
        27.651287492628626,
        30.39571351358609,
        15.434678487095226,
        0.0,
        14.219876933363384,
        17.38032220644945,
        23.229595347315033,
        18.637738596728948,
        38.86745167875043,
    ],
    [
        29.0363720185563,
        29.88243631299161,
        36.05471536429042,
        23.596016612979405,
        14.219876933363384,
        0.0,
        19.33075528788256,
        24.150846776044933,
        29.402166586835055,
        39.00029358863852,
    ],
    [
        10.292647861459168,
        10.949908675418264,
        16.84952818330532,
        8.884103781473964,
        17.38032220644945,
        19.33075528788256,
        0.0,
        5.870851726964334,
        15.765167300095486,
        21.6223957969509,
    ],
    [
        4.893475247715052,
        5.801603226695184,
        13.740047307051022,
        12.337949586539896,
        23.229595347315033,
        24.150846776044933,
        5.870851726964334,
        0.0,
        18.2839492451713,
        15.770393146652994,
    ],
    [
        18.916217909508234,
        18.548342243985044,
        15.593168375926687,
        6.983308671396387,
        18.637738596728948,
        29.402166586835055,
        15.765167300095486,
        18.2839492451713,
        0.0,
        29.321631946397524,
    ],
    [
        11.722533855783912,
        11.521688244350297,
        15.415686815708211,
        25.782817922019298,
        38.86745167875043,
        39.00029358863852,
        21.6223957969509,
        15.770393146652994,
        29.321631946397524,
        0.0,
    ],
]
