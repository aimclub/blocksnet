Quickstart
==========
BlockNet quick start guide

.. |network_model| image:: https://i.ibb.co/khQDKLq/output.png


How to install
--------------
.. code::

pip install git+https://github.com/iduprojects/masterplanning

How to get cut blocks
----------------------------------------------------

An example of using the BlockNet library to extract addresses from the "Comment text" column in the sample_data.csv file.  

- **Step 1**. Input data fetch and parameters setting.
.. code:: python

   from masterplan_tools.method.blocks import CutParameters

   city_geometry = gpd.read_parquet(os.path.join(example_data_path, "city_geometry.parquet")).to_crs(local_crs)
   water_geometry = gpd.read_parquet(os.path.join(example_data_path, "water_geometry.parquet")).to_crs(local_crs)
   roads_geometry = gpd.read_parquet(os.path.join(example_data_path, "roads_geometry.parquet")).to_crs(local_crs)
   railways_geometry = gpd.read_parquet(os.path.join(example_data_path, "railways_geometry.parquet")).to_crs(local_crs)

   cut_params = CutParameters(
   city=city_geometry,
   water=water_geometry,
   roads=roads_geometry,
   railways=railways_geometry
)


- **Step 2**. To improve our method we should use land use filtering. If we don't set landuse parameters, no LU filtering will be applied to the blocks. 
.. code:: python

   from masterplan_tools.method.blocks import LandUseParameters

   no_development = gpd.read_file(os.path.join(example_data_path, "no_development_pzz.geojson"), mask=city_geometry.to_crs(4326)).to_crs(local_crs)
   no_development = no_development[no_development['RAYON']=='Василеостровский']
   landuse = gpd.read_file(os.path.join(example_data_path, "landuse_zone_pzz.geojson"), mask=city_geometry.to_crs(4326)).to_crs(local_crs)
   buildings_geom = gpd.read_file(os.path.join(example_data_path, "buildings_blocks.geojson"), mask=city_geometry.to_crs(4326)).to_crs(local_crs)

   lu_params = LandUseParameters(
   no_development=no_development,
   landuse=landuse,
   buildings=buildings_geom
   )
   

- **Step 3**. To generate city blocks GeoDataFrame we use the `BlockCutter` class. 

.. code:: python

   from masterplan_tools.method import BlocksCutter

   blocks = BlocksCutter(
   cut_parameters=cut_params, 
   lu_parameters=lu_params,
   ).get_blocks()

   blocks.to_gdf().head()

There are three landuse tags in the blocks gdf:
  - 'no_dev_area' -- according to th no_debelopment gdf and cutoff without any buildings or specified / selected landuse types;
  - 'selected_area' -- according to the landuse gdf. We separate theese polygons since they have specified landuse types;
  - 'buildings' -- there are polygons that have buildings landuse type. 

In further calculations we will use the in the following steps:
 - Only 'buildings' -- to find clusters of buildings in big polygons;
 - All of them while calculating the accessibility times among city blocks;
 - All of them except 'no_dev_area' while optimizing the development of new facilities.

How to сreate CityModel
----------------------------------------------------
We use the results from our previous examples, but you can use your own prepared GeoDataFrames.

- **Step 1**. Load aggregated info we have and data required for service graphs creation.
.. code:: python

   aggregated_blocks = gpd.read_parquet(os.path.join(example_data_path, "data_getter_blocks.parquet"))
   accessibility_matrix = pd.read_pickle(os.path.join(example_data_path, "data_getter_matrix.pickle"))

   schools = gpd.read_parquet(os.path.join(example_data_path, "schools.parquet"))
   kindergartens = gpd.read_parquet(os.path.join(example_data_path, "kindergartens.parquet"))
   recreational_areas = gpd.read_parquet(os.path.join(example_data_path, "recreational_areas.parquet")).rename_geometry('geometry')
   hospitals = gpd.read_file(os.path.join(example_data_path, "hospitals.geojson"))
   pharmacies = gpd.read_file(os.path.join(example_data_path, "pharmacies.geojson"))
   policlinics = gpd.read_file(os.path.join(example_data_path, "policlinics.geojson"))

   services = {"schools": schools, "kindergartens": kindergartens, "recreational_areas": recreational_areas,
               "hospitals": hospitals, "pharmacies": pharmacies, "policlinics": policlinics}

- **Step 2**. Creation of a city model
.. code:: python

   from masterplan_tools import CityModel

   city_model = CityModel(
   blocks=aggregated_blocks, 
   accessibility_matrix=accessibility_matrix, 
   services=services
   )

   city_model.visualize()

| |network_model|