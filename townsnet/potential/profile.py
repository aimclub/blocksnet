from enum import Enum

class Profile(Enum):
  RESIDENTIAL_INDIVIDUAL = 'Жилая застройка - ИЖС'
  RESIDENTIAL_LOWRISE = 'Жилая застройка - Малоэтажная'
  RESIDENTIAL_MIDRISE = 'Жилая застройка - Среднеэтажная'
  RESIDENTIAL_MULTISTOREY = 'Жилая застройка - Многоэтажная'
  BUSINESS = 'Общественно-деловая'
  RECREATION = 'Рекреационная'
  SPECIAL = 'Специального назначения'
  INDUSTRIAL = 'Промышленная'
  AGRICULTURE = 'Сельско-хозяйственная'
  TRANSPORT = 'Транспортная инженерная'