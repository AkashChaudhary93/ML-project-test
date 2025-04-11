import pandas as pd
import numpy as np
from faker import Faker
import random
from faker.providers import BaseProvider

# Initialize Faker and custom provider
fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

class EngineeringProvider(BaseProvider):
    def engineering_field(self):
        fields = ['Computer Science', 'Mechanical', 'Electrical', 'Civil', 'Electronics']
        weights = [0.45, 0.18, 0.15, 0.12, 0.10]  # More realistic distribution
        return random.choices(fields, weights=weights, k=1)[0]
    
    def college_tier(self, jee_rank, twelfth_percent):
        def logistic_prob(offset, scale, x):
            return 1 / (1 + np.exp(-(offset + scale*x)))
        
        # Tier probabilities
        prob_top = logistic_prob(offset=2.5, scale=-0.00004, x=jee_rank) * logistic_prob(offset=-8, scale=0.15, x=twelfth_percent)
        prob_mid = logistic_prob(offset=1.8, scale=-0.00003, x=jee_rank) * logistic_prob(offset=-7, scale=0.13, x=twelfth_percent)
        prob_nit = logistic_prob(offset=1.2, scale=-0.00002, x=jee_rank) * logistic_prob(offset=-6, scale=0.11, x=twelfth_percent)
        prob_state = logistic_prob(offset=0.8, scale=-0.00001, x=jee_rank) * logistic_prob(offset=-5, scale=0.09, x=twelfth_percent)
        
        tiers = ['Top IIT', 'Mid IIT', 'NIT', 'State College', 'Private College']
        probs = [prob_top, prob_mid, prob_nit, prob_state, 1-sum([prob_top, prob_mid, prob_nit, prob_state])]
        
        return random.choices(tiers, weights=probs, k=1)[0]

fake.add_provider(EngineeringProvider)

# Generate synthetic dataset
num_records = 50000
data = []

for _ in range(num_records):
    # Academic scores with correlation
    tenth_percent = np.random.beta(a=2, b=1.5) * 40 + 60
    tenth_percent = round(np.clip(tenth_percent, 60, 100), 1)
    
    twelfth_percent = np.clip(tenth_percent * 0.9 + np.random.normal(5, 7), 60, 100)
    twelfth_percent = round(twelfth_percent, 1)
    
    # JEE Rank with realistic distribution
    jee_rank = int(np.exp(np.random.normal(8.5, 1.2)))
    jee_rank = max(1, min(jee_rank, 250000))
    
    # College tier assignment
    college_tier = fake.college_tier(jee_rank, twelfth_percent)
    
    # Personal details
    gender = random.choice(['Male', 'Female'])
    engineering_field = fake.engineering_field()
    
    # Experience based on college tier
    exp_params = {
        'Top IIT': {'lambda': 0.8, 'max_exp': 8},
        'Mid IIT': {'lambda': 0.7, 'max_exp': 7},
        'NIT': {'lambda': 0.6, 'max_exp': 6},
        'State College': {'lambda': 0.5, 'max_exp': 5},
        'Private College': {'lambda': 0.4, 'max_exp': 4}
    }
    params = exp_params[college_tier]
    experience = min(np.random.poisson(params['lambda']) + np.random.binomial(1, 0.2), params['max_exp'])
    
    # Enhanced salary calculation
    base_salaries = {
        'Top IIT': np.random.lognormal(mean=13.8, sigma=0.12),
        'Mid IIT': np.random.lognormal(mean=13.2, sigma=0.15),
        'NIT': np.random.lognormal(mean=12.8, sigma=0.18),
        'State College': np.random.lognormal(mean=12.2, sigma=0.2),
        'Private College': np.random.lognormal(mean=11.8, sigma=0.25)
    }
    
    field_multipliers = {
        'Computer Science': 1.7 + 0.1*(college_tier in ['Top IIT', 'Mid IIT']),
        'Electronics': 1.4,
        'Electrical': 1.2,
        'Mechanical': 1.0,
        'Civil': 0.95
    }
    
    salary = base_salaries[college_tier] * field_multipliers[engineering_field] * (1 + 0.08*experience)
    salary = int(np.clip(salary + np.random.normal(0, 50000), 300000, 2500000))  # Realistic noise
    
    data.append([
        tenth_percent,
        twelfth_percent,
        jee_rank,
        gender,
        experience,
        engineering_field,
        college_tier,
        salary
    ])

# Create DataFrame and save
columns = ['10th_percent', '12th_percent', 'jee_rank', 'gender', 
           'experience', 'engineering_field', 'college_tier', 'salary']
df = pd.DataFrame(data, columns=columns)
df.to_csv('engineering_data1.csv', index=False)

print(f"Generated {len(df)} records")
print(df.describe())