import pandas as pd
import numpy as np
from faker import Faker
import random
from faker.providers import BaseProvider
from scipy.stats import skewnorm

# Initialize Faker and custom provider
fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

class EngineeringProvider(BaseProvider):
    def engineering_field(self):
        fields = ['Computer Science', 'Mechanical', 'Electrical', 'Civil', 'Electronics']
        # Realistic distribution with regional variations
        weights = [0.38, 0.22, 0.18, 0.12, 0.10]  
        return random.choices(fields, weights=weights, k=1)[0]
    
    def college_tier(self, jee_rank, twelfth_percent):
        def sigmoid(x, offset, scale):
            return 1 / (1 + np.exp(-(x - offset)/scale))
        
        # Real-world admission probabilities with overlap
        top_prob = sigmoid(twelfth_percent, 92, 2) * sigmoid(1/(jee_rank/1000), 0.8, 0.2)
        mid_prob = sigmoid(twelfth_percent, 85, 3) * sigmoid(1/(jee_rank/1000), 0.6, 0.3)
        nit_prob = sigmoid(twelfth_percent, 75, 5) * sigmoid(1/(jee_rank/1000), 0.4, 0.4)
        state_prob = sigmoid(twelfth_percent, 65, 7) * (1 - sigmoid(jee_rank, 150000, 50000))
        
        probs = np.array([top_prob, mid_prob, nit_prob, state_prob, 0.1])
        probs /= probs.sum()  # Normalize
        
        tiers = ['Top IIT', 'Mid IIT', 'NIT', 'State College', 'Private College']
        return random.choices(tiers, weights=probs, k=1)[0]

fake.add_provider(EngineeringProvider)

def generate_academic_scores():
    # Realistic correlated scores with regional bias
    base_score = skewnorm.rvs(4, loc=65, scale=15)
    tenth = np.clip(base_score + np.random.normal(5, 3), 60, 98)
    twelfth = np.clip(tenth * 0.9 + np.random.normal(7, 4), 60, 97)
    return round(tenth, 1), round(twelfth, 1)

def generate_jee_rank(twelfth):
    # Realistic rank distribution with top rankers
    if twelfth > 95:
        rank = np.random.chisquare(5, 1)[0] * 100
    elif twelfth > 85:
        rank = np.random.exponential(5000)
    else:
        rank = np.random.lognormal(10, 1.5)
    return max(1, min(int(rank), 250000))

def calculate_salary(college_tier, field, experience, gender):
    # Base salaries with realistic distributions (in ₹)
    base_salaries = {
        'Top IIT': skewnorm.rvs(-4, loc=18, scale=5),          # 12-25 LPA
        'Mid IIT': skewnorm.rvs(-3, loc=14, scale=4),          # 8-18 LPA
        'NIT': skewnorm.rvs(-2, loc=10, scale=3),              # 6-14 LPA
        'State College': skewnorm.rvs(-1, loc=6, scale=2.5),   # 3-10 LPA
        'Private College': skewnorm.rvs(0, loc=4, scale=2)     # 2-8 LPA
    }
    
    # Field multipliers with overlap
    field_factors = {
        'Computer Science': 1.6 + np.random.uniform(-0.1, 0.2),
        'Electronics': 1.3 + np.random.uniform(-0.05, 0.1),
        'Electrical': 1.1 + np.random.uniform(-0.05, 0.05),
        'Mechanical': 1.0 + np.random.uniform(-0.1, 0.1),
        'Civil': 0.9 + np.random.uniform(-0.05, 0.15)
    }
    
    # Experience curve with diminishing returns
    exp_factor = 1 + (0.1 * experience) - (0.003 * experience**2)
    
    # Gender bias (5-15% penalty for female)
    gender_bias = 1 - np.random.choice([0, 0.05, 0.1, 0.15], p=[0.7, 0.15, 0.1, 0.05]) if gender == 'Female' else 1
    
    # Company size effect
    company_size = np.random.choice([0.8, 1.0, 1.2], p=[0.3, 0.5, 0.2])
    
    # Final calculation with realistic noise
    salary = (base_salaries[college_tier] * 
             field_factors[field] * 
             exp_factor * 
             gender_bias * 
             company_size + 
             np.random.normal(0, 0.5))
    
    return int(np.clip(salary * 100000, 250000, 3000000))

# Generate synthetic dataset
num_records = 50000
data = []

for _ in range(num_records):
    # Academic scores
    tenth, twelfth = generate_academic_scores()
    
    # JEE Rank
    jee_rank = generate_jee_rank(twelfth)
    
    # College tier
    college_tier = fake.college_tier(jee_rank, twelfth)
    
    # Personal details
    gender = random.choices(['Male', 'Female'], weights=[0.65, 0.35])[0]
    field = fake.engineering_field()
    
    # Experience with realistic career progression
    base_exp = np.random.poisson(2 if college_tier in ['Top IIT', 'Mid IIT'] else 1.5)
    experience = min(base_exp + np.random.geometric(0.3), 10)
    
    # Salary calculation
    salary = calculate_salary(college_tier, field, experience, gender)
    
    data.append([
        tenth, twelfth, jee_rank, gender, 
        experience, field, college_tier, salary
    ])

# Create DataFrame with realistic noise
columns = ['10th_percent', '12th_percent', 'jee_rank', 'gender',
           'experience', 'engineering_field', 'college_tier', 'salary']
df = pd.DataFrame(data, columns=columns)

# Add real-world noise
df['10th_percent'] = df['10th_percent'].apply(lambda x: min(100, x + np.random.randint(-2, 3)))
df['12th_percent'] = df['12th_percent'].apply(lambda x: min(100, x + np.random.randint(-3, 4)))
df['jee_rank'] = df['jee_rank'].apply(lambda x: max(1, x + np.random.randint(-500, 500)))

# Save dataset
df.to_csv('realistic_engineering_data.csv', index=False)

print(f"Generated {len(df)} records")
print("Data Statistics:")
print(df.describe())
print("\nCollege Tier Distribution:")
print(df['college_tier'].value_counts(normalize=True))
print("\nAverage Salary by Tier:")
print(df.groupby('college_tier')['salary'].mean())