import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import mne
import tensorflow as tf
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

# Placeholder for function to verify user - in practice, replace with database lookup
def check_user(username, password):
    # Dummy user database - in practice, replace with real user lookup and secure password handling
    user_db = {
        "user1": "1234",  
        "guest": "1234",
    }
    # Hash the password and compare it to the stored one
    # hashed_password = sha256(password.encode()).hexdigest()
    return user_db.get(username) == password

# Logout function
def logout():
    st.session_state['authentication_status'] = None
    st.experimental_rerun()

# Authentication state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

# Login page
if st.session_state['authentication_status'] is not True:
    company_name = "HEVA"
    logo_path = "Logo.png"

    col1, col2 = st.columns([0.2, 1])

    # Add the logo to the first column
    col1.image(logo_path, width=80)

    # Add the company name to the second column
    col2.title(company_name)
    st.title("Login to HEVA")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if check_user(username, password):
                st.session_state['authentication_status'] = True
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password.")
else:
    if st.button('Logout'):
        logout()
    mapping_dict = {
        "Schizophrenia": "schizophrenia_v1.h5",
        "Frontotemporal Dementia": "frontotemporalDementia_v1.h5",
        "Alzhiemers": "Alzheimers_vS.h5",
        "Parkinsons": "Parkinson_v1.h5",
        "Depression": "Depression_v1.h5",
        "Multi Modal": "multiModal_1_v5.h5"
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
        "Select a Model", ["Multi Modal", "Schizophrenia", "Frontotemporal Dementia", "Alzhiemers", "Parkinsons", "Depression"]
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

    def safe_get(dct, keys, default='Unknown'):
        """Safely get a value from a nested dictionary or return a default if not found."""
        for key in keys:
            dct = dct.get(key)
            if dct is None:
                return default
        return dct

    def display_file_info(data):
        st.title("EEG File Information")
        with st.expander("File Information", expanded=True):
            try:
                info = data.info

                file_data = {
                    "Measurement date": info.get('meas_date').strftime("%B %d, %Y %H:%M:%S GMT") if info.get('meas_date') else "Unknown",
                    "Experimenter": info.get('experimenter', 'Unknown'),
                    "Participant": safe_get(info, ['subject_info', 'id'], "Unknown"),
                    "Good channels": f"{len(mne.pick_types(info, eeg=True, stim=False))} EEG, {len(mne.pick_types(info, stim=True))} Stimulus",
                    "Bad channels": ', '.join(info.get('bads', [])) or "None",
                    "EOG channels": len(mne.pick_types(info, eog=True)),
                    "ECG channels": len(mne.pick_types(info, ecg=True)),
                    "Sampling frequency": f"{info.get('sfreq', 'Unknown')} Hz",
                    "Highpass": f"{info.get('highpass', 'Unknown')} Hz",
                    "Lowpass": f"{info.get('lowpass', 'Unknown')} Hz",
                    "Filenames": safe_get(info, ['filenames'], "Unknown"),
                    "Duration": f"{data.times[-1] / 60:.2f} minutes" if data.times.size else "Unknown"
                }
                for key, value in file_data.items():
                    col1, col2 = st.columns([2, 3])
                    col1.metric(label=key, value="")
                    col2.write(value)

                channel_names = info['ch_names']
                if channel_names:
                    st.subheader('Channel Names')
                    st.write(f"Total number of channels: {len(channel_names)}")
                    st.text(", ".join(channel_names))

            except Exception as err:
                st.error(f'Error occurred while reading file information: {err}')
            
        # EEG Data Visualization
        st.subheader("EEG Data Visualization")
        try:
            with st.expander("Raw EEG Data", expanded=False):
                # Sliders for plot dimensions
                width = st.slider("Select plot width:", 5, 40, 20, key='raw_width')
                height = st.slider("Select plot height:", 5, 40, 10, key='raw_height')
                fig, ax = plt.subplots(figsize=(width, height))
                data.plot(n_channels=30, title='Raw EEG Data', show=False)
                mne_fig = plt.gcf()
                mne_fig.set_size_inches(width, height)
                st.pyplot(mne_fig)
            
            # Filtering section
            with st.expander("Filtering", expanded=False):
                # High-pass and Low-pass filter
                data.filter(0.5, 50)

                # Slider for plot width and height
                plot_width = st.slider("Select plot width:", 10, 40, 10, key='filter_width')
                plot_height = st.slider("Select plot height:", 10, 40, 10, key='filter_heigh1')

            # 20reate filtered EEG plot
                fig = data.plot(n_channels=15, title="Filtered EEG Data", scalings={"eeg": 75e-6}, duration=10, start=10, show=False)
                fig.set_size_inches(plot_width, plot_height)  # Adjust the size of the figure
                st.pyplot(fig)

                # Frequency bands
                bands = {
                    'Delta': (0.5, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 13),
                    'Beta': (13, 30),
                    'Gamma': (30, 45)
                }

                # Create power spectral density plot
                fig2, ax = plt.subplots()
                fig2.set_size_inches(plot_width, plot_height)  # Adjust the size of the figure
                data.plot_psd(fmax=50, ax=ax, show=False, average=True)
                for band, freq in bands.items():
                    ax.axvspan(*freq, color='orange', alpha=0.4, label=f'{band} {freq[0]}-{freq[1]} Hz')
                st.pyplot(fig2)
                
            with st.expander("Frequency Bands", expanded=False):
                # Sliders for plot width and height
                plot_width = st.slider("Select plot width:", 5, 20, 10, key='band_width')
                plot_height = st.slider("Select plot height:", 5, 20, 5, key='band_height')

                for band, freq in bands.items():
                    fmin, fmax = freq
                    fig, ax = plt.subplots(figsize=(plot_width, plot_height))  # Adjust figure size based on slider values
                    data.plot_psd(fmin=fmin, fmax=fmax, show=False, average=True, ax=ax)
                    ax.axvspan(fmin, fmax, color='orange', alpha=0.4, label=f'{band} {fmin}-{fmax} Hz')
                    ax.set_title(f'{band} Power')
                    ax.legend(loc='upper right')
                    st.pyplot(fig)
            
            # Bandpass filtering
            with st.expander("Bandpass Filtering", expanded=False):
                data.filter(l_freq=1.0, h_freq=30.0)
                width = st.slider("Select plot width:", 5, 20, 10)
                height = st.slider("Select plot height:", 5, 20, 5)
                fig, ax = plt.subplots(figsize=(width, height))
                data.plot_psd(fmax=30, ax=ax, show=False)
                st.pyplot(fig)

            with st.expander("Power Spectral Density", expanded=False):
                # Add sliders for plot width and height
                plot_width = st.slider("Select plot width:", 5, 20, 10, key='psd_width')
                plot_height = st.slider("Select plot height:", 5, 20, 5, key='psd_height')

                # Set annotations for events
                annotations = mne.Annotations(onset=[10, 20, 30], duration=[1, 2, 1], description=['Event 1', 'Event 2', 'Event 3'])
                data.set_annotations(annotations)

                # Temporarily change the default figure size for plotting
                with plt.rc_context({'figure.figsize': (plot_width, plot_height)}):
                    fig4 = data.plot(start=5, duration=5, color='b', event_color='r', show=False)
                
                st.pyplot(fig4)

        except Exception as e:
            st.error(f'An error occurred during EEG data visualization: {e}')


    def create_pdf_report(data, bands):
        # Create a PDF buffer in memory
        pdf_buffer = BytesIO()
        pdf_pages = PdfPages(pdf_buffer)
        
        # EEG Data Visualization plots
        fig = data.plot(n_channels=30, title='Raw EEG Data', show=False)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)

        # Filtering section
        data.filter(0.5, 50)
        fig = data.plot(n_channels=15, title="Filtered EEG Data", scalings={"eeg": 75e-6}, duration=10, start=10, show=False)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)

        # Frequency bands
        for band, freq in bands.items():
            fmin, fmax = freq
            fig, ax = plt.subplots()
            data.plot_psd(fmin=fmin, fmax=fmax, show=False, average=True, ax=ax)
            ax.axvspan(fmin, fmax, color='orange', alpha=0.4, label=f'{band} {fmin}-{fmax} Hz')
            ax.set_title(f'{band} Power')
            ax.legend(loc='upper right')
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)
        
        # Bandpass filtering
        data.filter(l_freq=1.0, h_freq=30.0)
        fig, ax = plt.subplots()
        data.plot_psd(fmax=30, ax=ax, show=False)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)

        # Power Spectral Density
        annotations = mne.Annotations(onset=[10, 20, 30], duration=[1, 2, 1], description=['Event 1', 'Event 2', 'Event 3'])
        data.set_annotations(annotations)
        fig = data.plot(start=5, duration=5, color='b', event_color='r', show=False)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)

        # Close the PDF and save to a buffer
        pdf_pages.close()
        pdf_buffer.seek(0)
        return pdf_buffer
    
    def generate_file(data):
        # Information extraction and plotting code goes here
        # For brevity, this is left as a placeholder comment

        # Define frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }

        if st.button("Generate PDF Report"):
            pdf_buffer = create_pdf_report(data, bands)
            st.download_button(
                label="Download EEG Report as PDF",
                data=pdf_buffer,
                file_name="eeg_report.pdf",
                mime="application/pdf"
            )

    if uploaded_file is not None:
        with st.spinner("Reading & Processing EEG data..."):
            model = load_selected_model(mapping_dict[selected_model_name])
            
            file_name = os.path.basename(uploaded_file.name)
            # st.write(f"File Name: {file_name}")
            # st.write(f"File Size: {uploaded_file.size} bytes")
            
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

            if selected_model_name == 'Multi Modal':
                X = data_processing_multi_modal_1(raw)
                Y = model.predict(X)

                # Create a dictionary for class labels
                class_labels = {
                    0: "Healthy",
                    1: "Schizophrenic",
                    2: "Parkinson's",
                    3: "Frontotemporal",
                    4: "Depressed",
                    5: "Alzheimer's"
                }

                # Get the predicted class index and label
                predicted_class_index = np.argmax(Y)
                predicted_class_label = class_labels[predicted_class_index]

                # Display the predictions using a progress bar for visual effect
                st.write("## Prediction Probabilities")
                for index, (class_label, probability) in enumerate(zip(class_labels.values(), Y[0])):
                    percentage = np.round(probability * 100, 3)
                    st.write(f"Chances of {class_label}: ", percentage, "%")
                    if class_label == "Healthy":
                        # Green color for healthy
                        st.markdown(f"<progress value='{int(percentage)}' max='100' style='width: 100%; height: 20px; color: green;'></progress>", unsafe_allow_html=True)
                    else:
                        # Red color for other conditions
                        st.markdown(f"<progress value='{int(percentage)}' max='100' style='width: 100%; height: 20px; color: red;'></progress>", unsafe_allow_html=True)

                    if index == predicted_class_index:
                        # Bold and color the most likely diagnosis
                        color = "green" if class_label == "Healthy" else "red"
                        st.markdown(f"<h3 style='color: {color};'>Most likely diagnosis: {class_label}</h3>", unsafe_allow_html=True)

                # Emphasize the predicted class
                color = "green" if predicted_class_label == "Healthy" else "red"
                st.markdown(f"<h2 style='color: {color};'>Patient may be: <strong>{predicted_class_label}</strong></h2>", unsafe_allow_html=True)


            else:
                X = data_processing_multi_modal_1(raw)
                Y = model.predict(X)
                probability = np.round(Y[0][0]*100, 3)

                st.write("## Health Assessment")

                # Check if the probability is above a certain threshold
                if probability > 40:
                    # Using markdown to change the text color to red for the warning
                    st.markdown(f"<h3 style='color:red;'>⚠️ Patient may be suffering from {selected_model_name} with a {probability}% probability.</h3>", unsafe_allow_html=True)
                    # Optionally, you can add a progress bar to visually represent the probability
                    st.progress(int(probability))
                    st.write(f"Confidence level: {probability}%")
                else:
                    # Using markdown to change the text color to green for healthy
                    st.markdown("<h3 style='color:green;'>✅ Patient is likely healthy.</h3>", unsafe_allow_html=True)



        # st.write(Y)
        with st.spinner("Displaying EEG data..."):
            display_file_info(raw)
            generate_file(raw)

        # st.write("**All the results are out of 1**")
        os.remove(file_name)



