## Title of the Project
Vehicle-Resale-Value-Prediction-and-Customer-Centric-Vehicle-Recommendation-System
## About

The primary aim of this project is to develop a machine learning model that accurately predicts the resale value of vehicles based on various features such as brand, model, age, mileage, and market trends. This project seeks to assist users in making informed decisions regarding the buying and selling of vehicles by providing predictive analytics. Additionally, the project aims to incorporate advanced techniques like collaborative filtering and content-based filtering for personalized vehicle recommendations, as well as integrating virtual voice recognition to enhance user interaction and accessibility.

## Features
- Implements advanced machine learning models such as Random Forest and Support Vector Regression for accurate resale value predictions.
- Personalized recommendation system using Collaborative Filtering and Content-Based Filtering techniques.
- Integration of Virtual Voice Recognition, allowing seamless, voice-activated vehicle searches and recommendations.
- High scalability, enabling the system to handle large datasets and growing user demands.
- Optimized performance, reducing time complexity in prediction and recommendation processes.
- Real-time predictions of car resale values based on dynamic market trends and customer preferences.

## Requirements
### Hard ware requirements:
```
●	Processor	:Multi-core CPU (Intel i5/i7 or AMD Ryzen 5/7)
●	Storage         : SSD (at least 256 GB)
●	GPU             : Optional (NVIDIA GTX 1660 or RTX 2060 for deep learning)
●	RAM             : Minimum 8 GB (16 GB preferred)
●	Keyboard        :110 keys enhanced
```
### Software requirements:
```
•	Operating System                   : Windows, Linux (Ubuntu), or macOS
•	 Programming Language              : Python
•	Machine Learning Libraries         : Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
•	 Development Environment           : Jupyter Notebook, Anaconda, PyCharm, or VS Code
•	 Vesion control                    : Git,Git hub
```
# System Architecture

![image](https://github.com/user-attachments/assets/a71e4b97-dc76-4fdd-ad5a-f71b660d3402)

The graphic provides a concise and clear depiction of all the entities integrated into the Car Resale Value Prediction and Personalized Vehicle Recommendation system. It illustrates how various actions and decisions are interconnected, offering a visual representation of the entire process flow. The diagram outlines the functional relationships between different entities within the system. The system architecture shown is clearly demonstrates that the input is provided by the customer in the form of vehicle preferences such as brand, model, and mileage. The system retrieves historical vehicle data from the database, which is then processed through a Random Forest-based prediction model to estimate the resale value of the car. Simultaneously, the depreciation rate is calculated based on the car’s features and market trends.
# Program :
```
//Read Data
import pandas as pd
data = pd.read_csv('car_data.csv’)
//Data Preprocessing
data.fillna(method='ffill', inplace=True)
data['Age'] = 2024 - data['Year']
data = pd.get_dummies(data, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'], drop_first=True)
features = data.drop(columns=['Price
target = data['Price']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features[['Kilometers_Driven', 'Engine']] = scaler.fit_transform(features[['Kilometers_Driven', 'Engine’]])
//Model Training
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor 
from sklearn.ensemble import RandomForestRegressor
 from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
// Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
// Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
// Train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
//Train the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

//Model Evaluation
# Evaluate the models on the test set
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Predictions and evaluation
y_pred_val = gb_model.predict(X_val)
# Calculate RMSE and R² score
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2 = r2_score(y_val, y_pred_val)
print(f'Validation RMSE: {rmse}')
print(f'Validation R² Score: {r2}')
//Resale value prediction
resale_values = pd.DataFrame({'Car': test_data['Engine'], 'Predicted_Resale_Value': y_pred_test})
# Calculate percentiles
best_cars_threshold = resale_values['Predicted_Resale_Value'].quantile(0.75)

def get_user_input_and_find_vehicle():
    # Get user input (can skip with empty input)
    name = input("Enter car name (or press Enter to skip): ")

    # Convert numeric inputs to float/int if provided, otherwise set to None
    try:
        engine = float(input("Enter maximum engine size in CC (or press Enter to skip): ") or None)
    except ValueError:
        engine = None

    try:
        distance_traveled = float(input("Enter maximum distance traveled in km (or press Enter to skip): ") or None)
    except ValueError:
        distance_traveled = None

    try:
        budget = float(input("Enter maximum budget in lakhs (or press Enter to skip): ") or None)

except ValueError:
        budget = None
    try:
        power = float(input("Enter minimum power in bhp (or press Enter to skip): ") or None)
    except ValueError:
        power = None
    try:
        seats = int(input("Enter number of seats (or press Enter to skip): ") or None)
    except ValueError:
        seats = None
    # Find vehicles based on input criteria
    available_vehicles = find_vehicles(
        train_data=train_data_processed,
        name=name,
        engine=engine,
        distance_traveled=distance_traveled,
        budget=budget,
        power=power,
        seats=seats
    )
    print("\nAvailable Vehicles Matching Your Criteria:")
    print(available_vehicles)
# Call the function to get user input and find vehicles
get_user_input_and_find_vehicle()

```
## Output

## Output1 - Head and Tail values for Train and test data:

![image](https://github.com/user-attachments/assets/76796600-e8c2-4c20-8764-bb3036872f4d)

![image](https://github.com/user-attachments/assets/f143f75d-f341-403e-93d1-4ab66f650075)

## Output2 - Algorithms and Visualization pics:
![image](https://github.com/user-attachments/assets/aaa616ca-173a-4098-bee7-1cbc95126929)
![image](https://github.com/user-attachments/assets/843f43a7-0ca7-4b40-9904-85e5a9113857)
![image](https://github.com/user-attachments/assets/7b43ca2d-2ff9-49b5-8216-1c9938cfbae1)
![image](https://github.com/user-attachments/assets/d72d67d1-0187-4311-a1a0-5940667ce169)
![image](https://github.com/user-attachments/assets/163e9c23-e4f2-4c2f-8a5a-247d782016a6)
![image](https://github.com/user-attachments/assets/107bf944-6f22-4c9d-b092-8e5444e1565b)
![image](https://github.com/user-attachments/assets/b28fb47e-66f8-4db5-9191-81e023922c64)
## Output2 - Car resale Values:
![image](https://github.com/user-attachments/assets/992f29b1-58c7-4a40-a45f-c2c88d292c25)
![image](https://github.com/user-attachments/assets/5caf0c85-3f59-418f-ad4b-af4f59e766ad)




## Results and Impact

The system accurately predicted car resale values using machine learning algorithms such as Random Forest and Support Vector Regression.Personalized User Recommendations: Provided tailored vehicle recommendations based on Collaborative Filtering and Content-Based Filtering, enhancing user satisfaction.Virtual Voice Recognition: Successfully integrated Virtual Voice Recognition, enabling users to perform seamless, voice-activated car searches and recommendations.
### Impact :
Empowered buyers and sellers with data-driven insights, improving pricing strategies and purchase decisions.Optimized the used car market by minimizing price undervaluation and overpricing,The system is adaptable for future integration with real-time market data and predictive maintenance features
## Articles published / References 
```

[1] Smith, A., Johnson, B., & Lee, C. (2022). "Vehicle Resale Value Prediction Using Machine Learning Techniques." Journal of Automotive Data Science, 34(2), 112-125 
[2] Wang, Q., Zhang, Y., & Chen, F. (2020). "Deep Learning-Based Models for Predicting Vehicle Resale Values." Machine Learning for Automotive Industry, 18(1), 45-67.
[3] Zhang, L., Xu, M., & Wang, J. (2022). "Collaborative Filtering and Content-Based Filtering for Vehicle Recommendations." Intelligent Transportation Systems Journal, 41(4), 210-230. 
[4] Thomas, B., & Green, J. (2021). "Random Forest Models for Fleet Vehicle Resale Prediction." Journal of Predictive Analytics in Transportation, 29(3), 158-170. 
[5] Patel, R., & Kumar, A. (2020). "Advanced Machine Learning Models for Vehicle Price Prediction." International Journal of Data Science and Analytics, 14(3), 145-158


```


