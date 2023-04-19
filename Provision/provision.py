import pandas as pd
import numpy as np
import psycopg2 as pg
import geopandas as gpd
from tqdm.auto import tqdm 
tqdm.pandas()

engine = pg.connect("dbname='city_db_final' user='postgres' host='10.32.1.107' port='5432' password='postgres'")

GROUPS = ('Детский сад', 'Школа', 'Рекреационная зона')
SUB_GROUP = []