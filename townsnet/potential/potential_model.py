from .profile import Profile
from .indicator import Indicator

PROFILES_WEIGHTS = {
    Profile.RESIDENTIAL_INDIVIDUAL: {
        Indicator.POPULATION: 0.3, 
        Indicator.TRANSPORT: 0.2, 
        Indicator.ECOLOGY: 0.25, 
        Indicator.SOCIAL: 0.15, 
        Indicator.ENGINEERING: 0.1
    },
    Profile.RESIDENTIAL_LOWRISE: {
        Indicator.POPULATION: 0.3, 
        Indicator.TRANSPORT: 0.2, 
        Indicator.ECOLOGY: 0.25, 
        Indicator.SOCIAL: 0.15, 
        Indicator.ENGINEERING: 0.1
    },
    Profile.RESIDENTIAL_MIDRISE: {
        Indicator.POPULATION: 0.25, 
        Indicator.TRANSPORT: 0.25, 
        Indicator.ECOLOGY: 0.2, 
        Indicator.SOCIAL: 0.2, 
        Indicator.ENGINEERING: 0.1
    },
    Profile.RESIDENTIAL_MULTISTOREY: {
        Indicator.POPULATION: 0.2, 
        Indicator.TRANSPORT: 0.3, 
        Indicator.ECOLOGY: 0.2, 
        Indicator.SOCIAL: 0.2, 
        Indicator.ENGINEERING: 0.1
    },
    Profile.BUSINESS: {
        Indicator.POPULATION: 0.1, 
        Indicator.TRANSPORT: 0.35, 
        Indicator.ECOLOGY: 0.1, 
        Indicator.SOCIAL: 0.2, 
        Indicator.ENGINEERING: 0.25
    },
    Profile.RECREATION: {
        Indicator.POPULATION: 0.15, 
        Indicator.TRANSPORT: 0.15, 
        Indicator.ECOLOGY: 0.4, 
        Indicator.SOCIAL: 0.1, 
        Indicator.ENGINEERING: 0.2
    },
    Profile.SPECIAL: {
        Indicator.POPULATION: 0.1, 
        Indicator.TRANSPORT: 0.3, 
        Indicator.ECOLOGY: 0.1, 
        Indicator.SOCIAL: 0.15, 
        Indicator.ENGINEERING: 0.35
    },
    Profile.INDUSTRIAL: {
        Indicator.POPULATION: 0.05, 
        Indicator.TRANSPORT: 0.4, 
        Indicator.ECOLOGY: 0.1, 
        Indicator.SOCIAL: 0.1, 
        Indicator.ENGINEERING: 0.35
    },
    Profile.AGRICULTURE: {
        Indicator.POPULATION: 0.2, 
        Indicator.TRANSPORT: 0.15, 
        Indicator.ECOLOGY: 0.3, 
        Indicator.SOCIAL: 0.15, 
        Indicator.ENGINEERING: 0.2
    },
    Profile.TRANSPORT: {
        Indicator.POPULATION: 0.05, 
        Indicator.TRANSPORT: 0.45, 
        Indicator.ECOLOGY: 0.05, 
        Indicator.SOCIAL: 0.1, 
        Indicator.ENGINEERING: 0.35
    }
}

# PROFILES = {
#   Profile.RESIDENTIAL_INDIVIDUAL: {
#       "criteria": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 2, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 4, Indicator.ENGINEERING: 3},
#       "weights": {Indicator.POPULATION: 0, Indicator.TRANSPORT: 0, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 1, Indicator.ENGINEERING: 0}
#   },
#   Profile.RESIDENTIAL_LOWRISE: {
#       "criteria": {Indicator.POPULATION: 3, Indicator.TRANSPORT: 3, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 3, Indicator.ENGINEERING: 4},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 1, Indicator.ENGINEERING: 1}
#   },
#   Profile.RESIDENTIAL_MIDRISE: {
#       "criteria": {Indicator.POPULATION: 4, Indicator.TRANSPORT: 4, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 3, Indicator.ENGINEERING: 5},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 1, Indicator.ENGINEERING: 1}
#   },
#   Profile.RESIDENTIAL_MULTISTOREY: {
#       "criteria": {Indicator.POPULATION: 5, Indicator.TRANSPORT: 5, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 3, Indicator.ENGINEERING: 5},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 1, Indicator.ENGINEERING: 1}
#   },
#   Profile.BUSINESS: {
#       "criteria": {Indicator.POPULATION: 4, Indicator.TRANSPORT: 5, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 2, Indicator.ENGINEERING: 4},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 1}
#   },
#   Profile.RECREATION: {
#       "criteria": {Indicator.POPULATION: 0, Indicator.TRANSPORT: 0, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 0},
#       "weights": {Indicator.POPULATION: 0, Indicator.TRANSPORT: 0, Indicator.ECOLOGY: 0, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 0}
#   },
#   Profile.SPECIAL: {
#       "criteria": {Indicator.POPULATION: 0, Indicator.TRANSPORT: 3, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 2},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 1}
#   },
#   Profile.INDUSTRIAL: {
#       "criteria": {Indicator.POPULATION: 3, Indicator.TRANSPORT: 4, Indicator.ECOLOGY: 0, Indicator.SOCIAL: 2, Indicator.ENGINEERING: 4},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 0, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 1}
#   },
#   Profile.AGRICULTURE: {
#       "criteria": {Indicator.POPULATION: 3, Indicator.TRANSPORT: 4, Indicator.ECOLOGY: 4, Indicator.SOCIAL: 2, Indicator.ENGINEERING: 3},
#       "weights": {Indicator.POPULATION: 1, Indicator.TRANSPORT: 1, Indicator.ECOLOGY: 1, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 1}
#   },
#   Profile.TRANSPORT: {
#       "criteria": {Indicator.POPULATION: 2, Indicator.TRANSPORT: 2, Indicator.ECOLOGY: 0, Indicator.SOCIAL: 1, Indicator.ENGINEERING: 2},
#       "weights": {Indicator.POPULATION: 0, Indicator.TRANSPORT: 0, Indicator.ECOLOGY: 0, Indicator.SOCIAL: 0, Indicator.ENGINEERING: 0}
#   }
# }

class PotentialModel():

    def __init__(self, profiles_weights : dict[Profile, dict[Indicator, float]] = PROFILES_WEIGHTS):
        self.profiles_weights = profiles_weights

    # def _is_criterion_satisfied(profile_value, criterion_value):
    #     if isinstance(profile_value, tuple):
    #         return profile_value[0] <= criterion_value <= profile_value[1]
    #     return criterion_value >= profile_value

    # def _calculate_exceedance(profile_value, criterion_value):
    #     if isinstance(profile_value, tuple):
    #         if profile_value[0] <= criterion_value <= profile_value[1]:
    #             return criterion_value - profile_value[0]
    #         return 0
    #     return max(0, criterion_value - profile_value)

    def evaluate_potential(self, indicators_values : dict[Indicator, float], profile : Profile) -> float:
        
        profile_weights = self.profiles_weights[profile]
        return sum([weight * indicators_values[indicator] for indicator, weight in profile_weights.items()])

    def evaluate_potentials(self, indicators_values : dict[Indicator, float]) -> dict[Profile, float]:

        potential_scores = {}

        for profile in self.profiles_weights.keys():
            potential_scores[profile] = self.evaluate_potential(indicators_values, profile)

        return potential_scores

        # potential_scores = {}
        # for profile, data in PROFILES.items():
        #     criteria = data["criteria"]
        #     weights = data["weights"]
        #     potential = sum(
        #         _is_criterion_satisfied(criteria[criterion], value)
        #         for criterion, value in criteria_values.items()
        #     )
        #     weighted_score = sum(
        #         _calculate_exceedance(criteria.get(criterion, -1), value) * weights[criterion]
        #         for criterion, value in criteria_values.items()
        #     )
        #     potential_scores[profile.name] = {
        #         'potential': potential,
        #         'weighted_score': weighted_score
        #     }

        # return potential_scores