Quickstart
==========
BlocksNet quick start guide

.. |network_model| image:: https://i.ibb.co/khQDKLq/output.png


How to install
--------------
.. code::

   pip install git+https://github.com/iduprojects/blocksnet

Easy start with prepared data
-----------------------------
You can download the
`data
<https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP?usp=sharing>`_
we used in our study cases:

How to cut city blocks
----------------------------------------------------

``BlocksCutter`` class requires the following input data for blocks cutting process:

- **city** - city boundaries geometry. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).
- **water** - water objects geometries. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).
- **road** - road network geometries. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).
- **railways** - railways network geometries. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).
- **no_development** - territories with restricted development. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).
- **landuse** - basic landuse geometries. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).
- **buildings** - buildings geometries that are used for clustering inside blocks. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields: **id** (int, optional).

The result of cutting blocks is a ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fiels: **id** (int), **landuse** ("``no_dev_area``", "``selected_area``" or "``buildings``")

An example of using the BlocksNet library to cut city blocks for the city.

- **Step 1**. Input data fetch and parameters setting.
.. code:: python

   from blocksnet.method.blocks import CutParameters

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

   from blocksnet.method.blocks import LandUseParameters

   no_development = gpd.read_file(os.path.join(example_data_path, "no_development_pzz.geojson"), mask=city_geometry.to_crs(4326)).to_crs(local_crs)
   no_development = no_development[no_development['RAYON']=='Василеостровский']
   landuse = gpd.read_file(os.path.join(example_data_path, "landuse_zone_pzz.geojson"), mask=city_geometry.to_crs(4326)).to_crs(local_crs)
   buildings_geom = gpd.read_file(os.path.join(example_data_path, "buildings_blocks.geojson"), mask=city_geometry.to_crs(4326)).to_crs(local_crs)

   lu_params = LandUseParameters(
   no_development=no_development,
   landuse=landuse,
   buildings=buildings_geom
   )


- **Step 3**. To generate city blocks GeoDataFrame we use the ``BlocksCutter`` class.

.. code:: python

   from blocksnet.method import BlocksCutter

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

How aggregate blocks information and create the accessibility matrix
--------------------------------------------------------------------
The ``DataGetter`` class requires the following input data to aggregate blocks info:

* **blocks** - cutted blocks from the ``BlocksCutter``. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fiels: **id** (int), **landuse** ("``no_dev_area``", "``selected_area``" or "``buildings``")
* **buildings** - buildings objects. ``GeoDataFrame``, containing Point geometries and following fields:

  * **population_balanced** (int) - total population of the building
  * **building_area** (float) - building area (in square meters)
  * **living_area** (float) - living area (in square meters)
  * **storeys_count** (int) - storeys count of the building
  * **is_living** (bool) - is building living
  * **living_area_pyatno** (float) - living area pyatno (in square meters)
  * **total_area** (float) - total building area (in square meters

* **greenings** - green areas objects. ``GeoDataFrame``, containing Point geometries and following fields: **current_green_area** (int, square meters), **current_green_capacity** (int).
* **parkings** - parkings objects. ``GeoDataFrame``, containing Point geometries and following fields: **current_parking_capacity** (int).

- **Step 1**. Load cutted blocks and initialize a ``DataGetter`` object.
.. code:: python

   from blocksnet.preprocessing import DataGetter, AggregateParameters

   blocks = gpd.read_parquet(os.path.join(example_data_path, "blocks_cutter_result.parquet")).to_crs(local_crs)
   getter = DataGetter(blocks=blocks)

- **Step 2**. Load buildings, greenings and parkings geometries and aggregate information with ``aggregate_block_info()``
.. code:: python

   buildings = gpd.read_parquet(os.path.join(example_data_path, "buildings.parquet"))
   greenings = gpd.read_parquet(os.path.join(example_data_path, "greenings.parquet")).rename_geometry('geometry')
   parkings = gpd.read_parquet(os.path.join(example_data_path, "parkings.parquet")).rename_geometry('geometry')

   aggr_params = AggregateParameters(
     buildings=buildings,
     greenings=greenings,
     parkings=parkings
   )

   aggregated_blocks = getter.aggregate_blocks_info(params=aggr_params)
   aggregated_blocks.to_gdf().head()

The accessibility matrix is created with intermodal ``nx.Graph`` from the
`CityGeoTools
<https://github.com/iduprojects/CityGeoTools>`_
library, imported as a GraphML object.

.. code:: python

   import networkx as nx

   transport_graph = nx.read_graphml(os.path.join(example_data_path, "new_graph.graphml.xml"))
   accessibility_matrix = getter.get_accessibility_matrix(transport_graph)
   accessibility_matrix.df.head()

How to сreate CityModel
----------------------------------------------------

We use the results from our previous examples, but you can use your own prepared GeoDataFrames. The ``CityModel`` class requires the following input data:

* **aggregated_blocks** - cutted and aggregated city blocks. ``GeoDataFrame``, containing Polygon or MultiPolygon geometries and following fields:

  * **landuse** ("``no_dev_area``", "``selected_area``" or "``buildings``").
  * **block_id** (int) - unique city block identifier.
  * **is_living** (bool) - is block living.
  * **current_population** (float) total population of the block.
  * **floors** (float) - Median storeys count of the buildings inside the block.
  * **current_living_area** (float) - Total living area of the block (in square meters).
  * **current_green_capacity** (float) - Total greenings capacity (in units).
  * **current_green_area** (float) - Total greenings area (in square meters).
  * **current_parking_capacity** (float) - Total parkings capacity (in units).
  * **current_industrial_area** (float) - Total industrial area of the block (in square meters).
  * **area** (int) - Total area of the block (in square meters).

* **accessibility_matrix** - accessibility matrix between city blocks. ``DataFrame`` containing distances between all the blocks (in minutes)
* **services** - services dict, where **key** is a service type name, and **value** is a ``GeoDataFrame``, containing Point geometries and following fields: **capacity** (int) - total service object capacity.


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

   from blocksnet import CityModel

   city_model = CityModel(
   blocks=aggregated_blocks,
   accessibility_matrix=accessibility_matrix,
   services=services
   )

   city_model.visualize()

| |network_model|
