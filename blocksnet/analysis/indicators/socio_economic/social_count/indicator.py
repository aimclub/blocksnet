from enum import Enum, unique


@unique
class SocialCountIndicator(Enum):
    # education
    KINDERGARTEN = "kindergarten"
    SCHOOL = "school"
    COLLEGE = "college"
    UNIVERSITY = "university"
    EXTRACURRICULAR = "extracurricular"

    # healthcare
    HOSPITAL = "hospital"
    POLYCLINIC = "polyclinic"
    AMBULANCE = "ambulance"
    SANATORIUM = "sanatorium"
    SPECIAL_MEDICAL = "special_medical"
    PREVENTIVE_MEDICAL = "preventive_medical"
    PHARMACY = "pharmacy"

    # sports
    GYM = "gym"
    SWIMMING_POOL = "swimming_pool"
    OUTDOOR_SPORTS = "outdoor_sports"
    STADIUM = "stadium"

    # social
    ORPHANAGE = "orphanage"
    NURSING_HOME = "nursing_home"
    SOCIAL_SERVICE_CENTER = "social_service_center"

    # service
    POST = "post"
    BANK = "bank"
    MULTIFUNCTIONAL_CENTER = "multifunctional_center"

    # leisure
    LIBRARY = "library"
    MUSEUM = "museum"
    THEATRE = "theatre"
    CULTURAL_CENTER = "cultural_center"
    CINEMA = "cinema"
    CONCERT_HALL = "concert_hall"
    # STADIUM = 'stadium'
    ICE_ARENA = "ice_arena"
    MALL = "mall"
    PARK = "park"
    BEACH = "beach"
    ECO_TRAIL = "eco_trail"

    # security
    FIRE_STATION = "fire_station"
    POLICE = "police"

    # tourism
    HOTEL = "hotel"
    HOSTEL = "hostel"
    TOURIST_BASE = "tourist_base"
    CATERING = "catering"
