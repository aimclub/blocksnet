BASIC_SERVICE_TYPES = {
  'education': [
    {
      'name': 'kindergarten',
      'name_ru': 'детский сад',
      'weight': 0.2,
      'accessibility': 7,
      'demand': 61,
      'osm_tags': {
        'amenity': 'kindergarten'
      }
    },
    {
      'name': 'school',
      'name_ru': 'школа',
      'weight': 0.2,
      'accessibility': 15,
      'demand': 120,
      'osm_tags': {
        'amenity': 'school'
      }
    },
  ],
  'healthcare': [
    {
      'name': 'health_center',
      'name_ru': 'ФАП / амбулатория',
      'weight': 0.2,
      'accessibility': 10,
      'demand': 13,
      'osm_tags': {
        'amenity': 'clinic'
      }
    },
    {
      'name': 'pharmacy',
      'name_ru': 'аптека',
      'weight': 0.2,
      'accessibility': 15,
      'demand': 15,
      'osm_tags': {
        'amenity': 'pharmacy'
      }
    },
  ],
  'commerce': [
    {
      'name': 'convenience',
      'name_ru': 'продуктовый магазин',
      'weight': 0.2,
      'accessibility': 5,
      'demand': 180,
      'osm_tags': {
        'shop': 'supermarket'
      }
    },
    {
      'name': 'houseware',
      'name_ru': 'хозяйственный магазин',
      'weight': 0.2,
      'accessibility': 5,
      'demand': 180,
      'osm_tags': {
        'shop': 'houseware'
      }
    },
  ],
  'catering': [
    {
      'name': 'cafe',
      'name_ru': 'кафе / кофейня',
      'weight': 0.15,
      'accessibility': 15,
      'demand': 25,
      'osm_tags': {
        'amenity': 'cafe'
      }
    },
  ],
  'leisure': [
    {
      'name': 'sports_hall',
      'name_ru': 'универсальный зал',
      'weight': 0.1,
      'accessibility': 15,
      'demand': 180,
      'osm_tags': {
        'leisure': 'sports_hall'
      }
    },
  ],
  'recreation': [
    {
      'name': 'playground',
      'name_ru': 'детская площадка',
      'weight': 0.2,
      'accessibility': 4,
      'demand': 2,
      'osm_tags': {
        'leisure': 'playground'
      }
    },
    {
      'name': 'park',
      'name_ru': 'сквер / бульвар / лесопарк',
      'weight': 0.2,
      'accessibility': 30,
      'demand': 150,
      'osm_tags': {
        'leisure': 'park'
      }
    },
  ],
  'sport': [
    {
      'name': 'fitness',
      'name_ru': 'воркаут / школьный спортзал',
      'weight': 0.15,
      'accessibility': 45,
      'demand': 10,
      'osm_tags': {
        'leisure': 'fitness_station'
      }
    },
  ],
  'service': [
    {
      'name': 'delivery',
      'name_ru': 'пункт доставки',
      'weight': 0.1,
      'accessibility': 15,
      'demand': 10,
      'osm_tags': {
        'shop': 'outpost'
      }
    },
    {
      'name': 'beauty',
      'name_ru': 'парикмахерская / салон красоты',
      'weight': 0.1,
      'accessibility': 15,
      'demand': 10,
      'osm_tags': {
        'shop': ['beauty', 'hairdresser'],
        'beauty': ['nails', 'tanning', 'spa', 'massage']
      }
    },
  ],
  'transport': [
    {
      'name': 'bus_stop',
      'name_ru': 'остановка ОТ',
      'weight': 0.2,
      'accessibility': 10,
      'demand': 500,
      'osm_tags': {
        'highway': 'bus_stop'
      }
    },
    {
      'name': 'parking',
      'name_ru': 'парковка',
      'weight': 0.2,
      'accessibility': 7,
      'demand': 378,
      'osm_tags': {
        'amenity': 'parking'
      }
    },
  ],
  'safeness': [
    {
      'name': 'police',
      'name_ru': 'участковый пункт полиции',
      'weight': 0.4,
      'accessibility': 10,
      'demand': 5,
      'osm_tags': {
        'amenity': 'police'
      }
    },
  ]
}