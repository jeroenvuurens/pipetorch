from .textcollection import TextCollection
from .dframe import DFrame
from .helper import path_shared, path_user

def read_torchtext(torchtext_function):
    try:
        return torchtext_function(root=path_shared() )
    except:
        return torchtext_function(root=path_user() )

def wine_quality():
    return DFrame.read_from_kaggle('uciml/red-wine-quality-cortez-et-al-2009')

def telco_churn():
    return DFrame.read_from_kaggle('blastchar/telco-customer-churn')

def movielens_ratings():
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    return DFrame.read_from_kaggle('odedgolden/movielens-1m-dataset', 'ratings.dat', sep='::', engine='python', encoding = "ISO-8859-1", names=COLS)

def movielens_movies():
    COLS = ['movie_id', 'title', 'genre']
    return DFrame.read_from_kaggle('odedgolden/movielens-1m-dataset', 'movies.dat', sep='::', engine='python', encoding = "ISO-8859-1", names=COLS)
    
def movielens_users():
    COLS = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    return DFrame.read_from_kaggle('odedgolden/movielens-1m-dataset', 'users.dat', sep='::', engine='python', encoding = "ISO-8859-1", names=COLS)
    
def dam_outflow(**kwargs):
    return DFrame.read_from_package('pipetorch', 'data/datasets/dam_outflow.csv', **kwargs)
    
def realestate(**kwargs):
    return DFrame.read_from_package('pipetorch', 'data/datasets/realestate.csv', **kwargs)
    
def heart_disease():
    return DFrame.read_from_kaggle('mlms1', 'train.csv', 'test_set.csv')
    
def boston_housing_prices():
    return DFrame.read_from_kaggle('fedesoriano/the-boston-houseprice-data')

def hotel_booking():
    return DFrame.read_from_kaggle('mojtaba142/hotel-booking')

def hotel():
    df = hotel_booking()
    return df[(df.ArrivalDateYear < 2017) | (df.ArrivalDateWeekNumber < 14)]

def hotel_test_orig():
    df = hotel_booking()
    return df[(df.ArrivalDateYear == 2017) & (df.ArrivalDateWeekNumber > 13)]

def hotel_test():
    return hotel_test_orig().drop(columns='IsCanceled')

def hotel_test_y():
    return hotel_test_orig()[['IsCanceled']]

def hotel_test_score(pred_y):
    return f1_score(hotel_test_y(), pred_y)

def iris():
    return DFrame.read_from_kaggle('uciml/iris').drop(columns='Id')

def bank_marketing():
    return DFrame.read_from_kaggle('janiobachmann/bank-marketing-dataset')

def auto_mpg():
    return DFrame.read_from_kaggle('uciml/autompg-dataset')

def big_mart_sales():
    return DFrame.read_from_kaggle('brijbhushannanda1979/bigmart-sales-data', train='Train.csv', test='Test.csv')

def advertising_sales():
    return DFrame.read_from_kaggle('yasserh/advertising-sales-dataset', 
                             index_col=0, header=0, 
                             names=['', 'TV', 'Radio', 'Newspaper', 'Sales'])

def titanic():
    return DFrame.read_from_kaggle('titanic', 'train.csv', 'test.csv')

def diamonds():
    return DFrame.read_from_kaggle('shivam2503/diamonds')

def indian_liver():
    return DFrame.read_from_kaggle('uciml/indian-liver-patient-records')

def ames_housing():
    return DFrame.read_from_kaggle('house-prices-advanced-regression-techniques', 'train.csv', 'test.csv')
    
def air_passengers():
    return DFrame.read_from_kaggle('rakannimer/air-passengers')

def rossmann_store_sales():
    return DFrame.read_from_kaggle('rossmann-store-sales', 'train.csv', 'test.csv')

def california():
    return DFrame.read_from_kaggle('camnugent/california-housing-prices')
    
def heart_disease(**kwargs):
    return DFrame.read_from_kaggle('johnsmith88/heart-disease-dataset')
    
def speeddate(**kwargs):
    return DFrame.read_from_kaggle('annavictoria/speed-dating-experiment')
    
def occupancy():
    return DFrame.read_from_kaggle('robmarkcole/occupancy-detection-data-set-uci', 'datatraining.txt', 'datatest.txt')

def ag_news(language='basic_english', min_freq=1, collate='pad'):
    from torchtext.datasets import AG_NEWS
    train_iter, test_iter = read_torchtext(AG_NEWS)
    tc = TextCollection.from_iter(train_iter, None, test_iter, min_freq=min_freq).collate(collate)
    return tc

def bbc_news(language='basic_english', min_freq=1, collate='pad'):
    return TextCollection.from_csv('/data/datasets/bbc-text.csv', language=language, min_freq=min_freq).collate(collate)
