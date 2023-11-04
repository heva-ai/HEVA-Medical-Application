import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import mne
import tensorflow as tf
from sklearn.decomposition import PCA


mapping_dict = {
    "Schizophrenia": "schizophrenia_v1.h5",
    "Frontotemporal Dementia": "frontotemporalDementia_v1.h5",
    "Alzhiemers": "Alzheimers_vS.h5",
    "Parkinsons": "Parkinson_v1.h5",
    "Depression": "Depression_v1.h5",
    "Multi Modal 1": "multiModal_1_v5.h5",
    "Multi Modal 2": "multiModal_2_v4.h5"
}

# Load the model based on the selection
def load_selected_model(model_name):
    model = tf.keras.models.load_model(f"Models/{model_name}")
    return model

company_name = "HEVA"
logo_path = "Logo.png"

col1, col2 = st.columns([0.2, 1])

# Add the logo to the first column
col1.image(logo_path, width=80)

# Add the company name to the second column
col2.title(company_name)

# Sidebar with dropdown for model selection
selected_model_name = st.sidebar.selectbox(
    "Select a Model", ["Schizophrenia", "Frontotemporal Dementia", "Alzhiemers", "Parkinsons", "Depression", "Multi Modal 1", "Multi Modal 2"]
)
st.title(f"Selected Model: {selected_model_name}")

# Main Content
uploaded_file = st.file_uploader("Choose a file to upload", type=["edf", "fif", 'bdf', 'set'], key="file")

def read_eeg_data(file_name, file_extension):
    """Load EEG data based on file extension."""
    if file_extension == ".edf":
        return mne.io.read_raw_edf(file_name, preload=True)
    elif file_extension == ".fif":
        return mne.io.read_raw_fif(file_name, preload=True)
    elif file_extension == ".set":
        return mne.io.read_raw_eeglab(file_name, preload=True)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def preprocess_data_new_arc(raw_data, desired_length=15000):
    """Preprocess the raw EEG data."""
    # Resample and convert to DataFrame
    raw_data.resample(250)
    df = raw_data.to_data_frame()

    # Normalize data
    df_stats = df.describe().transpose()
    df = (df - df_stats['mean']) / df_stats['std']

    # Zero-padding or truncating data to match desired_length
    if len(df) < desired_length:
        df = df.append([0]*(desired_length - len(df)))

    df = df.iloc[:desired_length]

    return df.values

def data_processing_multi_modal_1(raw):
    raw.resample(sfreq=250)
    data = raw.get_data()
    df = pd.DataFrame(data)
    df = df.transpose()
    info = df.values
    info = (info - np.mean(info))/np.std(info)
    pca = PCA(n_components=10)
    pca.fit(info)
    info = pca.transform(info)

    X = info[0:15000]
    X = np.reshape(X,[1,15000,10])

    return X

def reduce_dimensions(data, n_components=10):
    """Reduce data dimensions using PCA."""
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)

def data_processing_pipeline(raw, desired_length, n_components):
    processed_data = preprocess_data_new_arc(raw, desired_length)
    reduced_data = reduce_dimensions(processed_data, n_components)

    # Reshape for model input
    X = reduced_data.reshape([1, desired_length, n_components])

    return X

def display_file_info(data):
    # Display the title and info
    st.title("EEG Information")
    st.markdown("---")

    # Extract and display specific details from the raw info
    info = data.info
    try:
        data = {
            "Measurement date": info['meas_date'].strftime("%B %d, %Y %H:%M:%S GMT"),
            "Experimenter": info['experimenter'],
            "Participant": info['subject_info']['id'] if 'subject_info' in info and 'id' in info['subject_info'] else "Unknown",
            "Digitized points": "Not available",
            "Good channels": f"{len(info['chs']) - len(mne.pick_types(raw.info, eeg=False, stim=True))} EEG, {len(mne.pick_types(raw.info, stim=True))} Stimulus",
            "Bad channels": ', '.join(info['bads']) or "None",
            "EOG channels": len(mne.pick_types(raw.info, eog=True)),
            "ECG channels": len(mne.pick_types(raw.info, ecg=True)),
            "Sampling frequency": f"{info['sfreq']} Hz",
            "Highpass": f"{info['highpass']} Hz",
            "Lowpass": f"{info['lowpass']} Hz",
            "Filenames": uploaded_file,
            "Duration": str(raw.times[-1] / 60) + " minutes"
        }
        st.markdown("## File Information")
        for key, value in data.items():
            st.write(f"{key}: {value}")
    except Exception as Err:
        st.write('Error occurred while reading file information', Err)

    
    st.markdown("## Raw EEG Data")
    fig0 = raw.plot(n_channels=30, title='Raw EEG raw')
    st.pyplot(fig0)

    st.markdown("## High-Pass | Low-Pass Filter")
    # High-pass and Low-pass filter the raw to remove noise
    raw.filter(0.5, 50)
    fig = raw.plot(n_channels=15, title="Filtered EEG Data", scalings={"eeg": 75e-6}, duration=10, start=10, show=False)
    st.pyplot(fig)

    st.markdown("## Filtering According to 5 waves")
    # Define the frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    # Plot the power spectral density for each channel, highlighting the bands
    fig2 = raw.plot_psd(fmax=50, show=False, average=True)
    for band, freq in bands.items():
        plt.axvspan(*freq, color='orange', alpha=0.4, label=f'{band} {freq[0]}-{freq[1]} Hz')
    st.pyplot(fig2)

    st.markdown("## Invdividual Split")
    # Plot each frequency band using the function
    for band, freq in bands.items():
        fmin, fmax = freq
        fig, ax = plt.subplots()
        raw.plot_psd(fmin=fmin, fmax=fmax, show=False, average=True, ax=ax)
        ax.axvspan(fmin, fmax, color='orange', alpha=0.4, label=f'{band} {fmin}-{fmax} Hz')
        ax.set_title(f'{band} Power')
        ax.legend(loc='upper right')
        st.pyplot(fig)
    
    st.markdown("## Bandpass Filtering")
    raw.filter(l_freq=1.0, h_freq=30.0)
    width = st.slider("Select plot width:", min_value=5, max_value=20, value=10)
    height = st.slider("Select plot height:", min_value=5, max_value=20, value=5)
    fig, ax = plt.subplots(figsize=(width, height))
    raw.plot_psd(fmax=30, ax=ax, show=False)  # Using the show=False argument to prevent automatic display
    st.pyplot(fig)

    st.markdown("## Power Spectral Density")
    annotations = mne.Annotations(onset=[10, 20, 30], duration=[1, 2, 1], description=['Event 1', 'Event 2', 'Event 3'])
    raw.set_annotations(annotations)
    fig4 = raw.plot(start=5, duration=5, color='b', event_color='r')
    st.pyplot(fig4)



if uploaded_file is not None:
    with st.spinner("Reading & Processing EEG data..."):
        model = load_selected_model(mapping_dict[selected_model_name])
        
        file_name = os.path.basename(uploaded_file.name)
        st.write(f"File Name: {file_name}")
        st.write(f"File Size: {uploaded_file.size} bytes")
        
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension in [".edf"]:
            raw = mne.io.read_raw_edf(file_name, preload=True)
        elif file_extension in [".fif"]:
            raw = mne.io.read_raw_fif(file_name, preload=True)
        elif file_extension in [".bdf"]:
            raw = mne.io.read_raw_bdf(file_name, preload=True)
        elif file_extension == ".set":
            raw = mne.io.read_raw_eeglab(file_name, preload=True)
        

        if selected_model_name=='Multi Modal 2':
            X = data_processing_pipeline(raw, desired_length=250, n_components=8)
            Y = model.predict(X)
            st.write(f"Chances of Healthy = {np.round(Y[0][0]*100, 3)}%")
            st.write(f"Chances of Alcoholic = {np.round(Y[0][1]*100, 3)}%")
            st.write(f"Chances of Schizophrenic = {np.round(Y[0][2]*100, 3)}%")
            st.write(f"Chances of Depressed = {np.round(Y[0][3]*100, 3)}%")

            # Get the predicted class index
            predicted_class_index = np.argmax(Y)

            # Create a dictionary mapping from class index to class label
            class_labels = {
                0: "Healthy",
                1: "Alcoholic",
                2: "Schizophrenic",
                3: "Depressed",
            }

            # Get the predicted class label
            predicted_class_label = class_labels[predicted_class_index]
            st.write(f"## Patient maybe: {predicted_class_label}")
        
        elif selected_model_name=='Multi Modal 1':
            # X = data_processing_pipeline(raw, desired_length=15000, n_components=10)
            X = data_processing_multi_modal_1(raw)
            Y = model.predict(X)
            st.write(f"Chances of Healthy = {np.round(Y[0][0]*100, 3)}%")
            st.write(f"Chances of Schizophrenic = {np.round(Y[0][1]*100, 3)}%")
            st.write(f"Chances of Perkinsons = {np.round(Y[0][2]*100, 3)}%")
            st.write(f"Chances of Frontotemporal = {np.round(Y[0][3]*100, 3)}%")
            st.write(f"Chances of Depressed  = {np.round(Y[0][4]*100, 3)}%")
            st.write(f"Chances of Alzhiemers  = {np.round(Y[0][5]*100, 3)}%")

            # Get the predicted class index
            predicted_class_index = np.argmax(Y)

            # Create a dictionary mapping from class index to class label
            class_labels = {
                0: "Healthy",
                1: "Schizophrenic",
                2: "Perkinsons",
                3: "Frontotemporal",
                4: "Depressed",
                5: "Alzhiemers"
            }

            # Get the predicted class label
            predicted_class_label = class_labels[predicted_class_index]
            st.write(f"## Patient maybe: {predicted_class_label}")

        else:
            # X = data_processing_pipeline(raw, desired_length=15000, n_components=10)
            X = data_processing_multi_modal_1(raw)
            Y = model.predict(X)
            if np.round(Y[0][0]*100,3)>40:
                st.markdown(f'### Patient maybe suffering from {selected_model_name}: based on {np.round(Y[0][0]*100, 3)}% accuracy')
            else:
                st.write("Patient is healthy")


    # st.write(Y)
    with st.spinner("Displaying EEG data..."):
        display_file_info(raw)

    # st.write("**All the results are out of 1**")
    os.remove(file_name)


