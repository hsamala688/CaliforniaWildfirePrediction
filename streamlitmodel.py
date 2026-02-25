"""
This is where you guys are going to be building the specific RCF that is going to be used for predicting California wildfires.
For this you are going to not only need to use pandas, numpy, scikit-learn, but also joblib to product the
requisite pkl files that are needed for the model to be used in the streamlit app. Joblib is specifically
used to save the trained model as a pickle file and is industry standard as it is great for saving large
NumPy arrays which is what the model outputs.

As for the columns needed to be used in the model I recommend looking through the dataset that was uploaded
to the google drive and downloading it locally. When making any pushes be careful not to push the dataset as well.
I recommend using the following columns:
['fire', 'latitude', 'longitude', 'acq_hour', 'wx_tavg_c',
'wx_prcp_mm', 'wx_wspd_ms', 'lf_evc', 'lf_evh', 'EVT_FUEL_N']
feel free to use more, if you do please make sure to update the columns list accordingly.

Also once you have fully built out the model feel free to delete this block of comments, just make sure to include
the list of columns used in the model.

I leave it to you guys to divide the work equally between the two of you."""
