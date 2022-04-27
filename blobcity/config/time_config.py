import itertools
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing, Holt



class time_config:

    models={
        "ARIMA":[
            ARIMA,
            {
                "order":{'str':list(itertools.product(range(0, 3), range(0, 3), range(0, 3)))},
                'trend':{"str":['n','c','t','ct']}
            }
        ],
        "SARIMAX":[
            SARIMAX,
            {
                "order":{'str':list(itertools.product(range(0, 3), range(0, 3), range(0, 3)))},
                "seasonal_order":{'str':[(x[0], x[1], x[2], 12) for x in list(itertools.product(range(0, 3), range(0, 3), range(0, 3)))]},
                'trend':{"str":['n','c','t','ct']},
                "initialization":{"str":["approximate_diffuse"]}
            }
        ],
        "ExponentialSmoothing":[
            ExponentialSmoothing,
            {
                "seasonal":{"str":['add','mul',None]},
                "trend":{"str":['add','mul',None]},
                "initialization_method":{"str":['estimated','heuristic']}  
            }
        ]
    }
