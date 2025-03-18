import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from ase.io import read
from dscribe.descriptors import ValleOganov, SOAP, LMBTR, MBTR
from matminer.featurizers.structure import PartialRadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor  # For converting ASE Atoms to pymatgen Structure
from collections import deque, defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import combinations

st.title("Structure Descriptor Calculation & t-SNE Visualization")

# --- File Upload and Species Detection ---
uploaded_files = st.file_uploader(
    "Upload Structure Files (CIF, POSCAR, XYZ, etc.)",
    type=None,
    accept_multiple_files=True,
    key="uploaded_files"
)

if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")
    #if st.button("Remove Uploaded Files"):
    #    st.session_state.pop("uploaded_files", None)
    #    st.session_state.pop("species_list", None)
    #    try:
    #        st.experimental_rerun()
    #    except AttributeError:
    #        st.write("Please refresh the page manually.")
else:
    st.session_state.pop("species_list", None)

if uploaded_files:
    detected_species = set()
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        structure = read(uploaded_file.name)
        detected_species.update([atom.symbol for atom in structure])
    st.session_state["species_list"] = sorted(detected_species)

st.subheader("📊 Detected Atomic Species")
if "species_list" in st.session_state:
    species_list = st.session_state["species_list"]
    st.write(f"🧪 **{', '.join(species_list)}**")
else:
    st.write("🔹 No structures uploaded yet.")

# --- Descriptor Selection ---
st.divider()
st.subheader("⚙️ SET: ⟶ Select Descriptors")
descriptor_options = {
    "SOAP": SOAP,
    "ValleOganov": ValleOganov,
    "PRDF": PartialRadialDistributionFunction,
}
selected_descriptors = st.multiselect("⚙️ Choose Descriptors", list(descriptor_options.keys()))



# --- Live Descriptor Length Preview (placed just below the selected descriptors) ---
if uploaded_files and "species_list" in st.session_state and selected_descriptors:
    # Build preview parameters using default values.
    preview_params = {}
    for desc in selected_descriptors:
        if desc == "SOAP":
            species_str = st.session_state.get(f"{desc}_species", ", ".join(species_list))
            species = [s.strip() for s in species_str.split(",")]
            preview_params[desc] = {"species": species, "r_cut": 6.0, "n_max": 8, "l_max": 6}
        elif desc == "ValleOganov":
            species_str = st.session_state.get(f"{desc}_species", ", ".join(species_list))
            species = [s.strip() for s in species_str.split(",")]
            preview_params[desc] = {"species": species, "function": "distance", "sigma": 10 ** (-0.5), "n": 100,
                                    "r_cut": 8.0}
        elif desc == "PRDF":
            preview_params[desc] = {"cutoff": 20.0, "bin_size": 0.2}
        else:
            preview_params[desc] = {}
    preview_lengths = {}
    try:
        structure = read(uploaded_files[0].name)
        for desc in selected_descriptors:
            if desc == "PRDF":
                mg_structure = AseAtomsAdaptor.get_structure(structure)
                featurizer = descriptor_options[desc](**preview_params[desc])
                featurizer.fit([mg_structure])
                descriptor_values = featurizer.featurize(mg_structure)
            else:
                featurizer = descriptor_options[desc](**preview_params.get(desc, {}))
                descriptor_values = featurizer.create(structure)
            preview_lengths[desc] = (len(descriptor_values.flatten())
                                     if isinstance(descriptor_values, np.ndarray)
                                     else len(descriptor_values))
    except Exception as e:
        for desc in selected_descriptors:
            preview_lengths[desc] = f"Error: {e}"

# --- Descriptor Parameter Inputs ---
if uploaded_files and "species_list" in st.session_state:
    species_list = st.session_state["species_list"]
    descriptor_parameters = {}
    descriptor_lengths = {}

    for desc in selected_descriptors:
        st.subheader(f"⚙️ SET: ⟶ Parameters for {desc}")
        if desc == "SOAP":
            species_input = st.text_input(f"📊 {desc} - Species", ", ".join(species_list), key=f"{desc}_species",
                                          help="Comma-separated list of atomic species (e.g., H, O, C). \nIt is automatically filled by reading elements in input structures.")
            species = [s.strip() for s in species_input.split(",")]
            r_cut = st.number_input(f"⚙️ {desc} - Cutoff Radius (Å)", min_value=1.0, max_value=10.0, value=6.0,
                                    key=f"{desc}_r_cut",  help="Maximum distance to consider neighboring atoms (in Å).")
            n_max = st.number_input(f"⚙️ {desc} - n_max", min_value=1, max_value=20, value=8, step=1, key=f"{desc}_n_max",
                                    help="Maximum number of radial basis functions.")
            l_max = st.number_input(f"⚙️ {desc} - l_max", min_value=1, max_value=10, value=6, step=1, key=f"{desc}_l_max",
                                    help="Maximum number of spherical harmonics.")
            sigma = st.number_input(f"⚙️ {desc} - Sigma", min_value=0.01, max_value=2.0, value=1.0, step=0.01,
                                    key=f"{desc}_sigma",
                                    help="Standard deviation of the Gaussians used to expand atomic density.")

            rbf = st.selectbox(f"⚙️ {desc} - Radial Basis Function", ["gto", "polynomial"], index=0, key=f"{desc}_rbf",
                               help="Radial basis functions used for SOAP calculations.")

            # Weighting function dictionary
            weighting_function = st.selectbox(
                f"⚙️ {desc} - Weighting Function",
                ["None", "poly", "pow", "exp"],
                index=0,
                key=f"{desc}_weighting_function",
                help="Defines the weighting method for SOAP."
            )

            if weighting_function == "None":
                weighting = None
            else:
                weighting = {"function": weighting_function}

                # Get common parameters regardless of function
                c = st.number_input(
                    f"⚙️ {desc} - Weighting c", min_value=0.01, max_value=5.0, value=1.0, key=f"{desc}_weighting_c"
                )
                r0 = st.number_input(
                    f"⚙️ {desc} - Weighting r0", min_value=1.0, max_value=10.0, value=3.0, key=f"{desc}_weighting_r0"
                )

                if weighting_function == "poly":
                    m = st.number_input(
                        f"⚙️ {desc} - Weighting m", min_value=1, max_value=10, value=2, key=f"{desc}_weighting_m"
                    )
                    weighting.update({"c": c, "m": m, "r0": r0})

                elif weighting_function == "pow":
                    m = st.number_input(
                        f"⚙️ {desc} - Weighting m", min_value=1, max_value=10, value=2, key=f"{desc}_weighting_m"
                    )
                    weighting.update({"c": c, "m": m, "r0": r0})
                    d = st.number_input(
                        f"⚙️ {desc} - Weighting d", min_value=0.01, max_value=5.0, value=1.0, key=f"{desc}_weighting_d"
                    )
                    weighting["d"] = d
                    threshold = st.number_input(
                        f"⚙️ {desc} - Weighting Threshold", min_value=1e-6, max_value=1e-1, value=1e-2,
                        key=f"{desc}_weighting_threshold"
                    )
                    weighting["threshold"] = threshold

                elif weighting_function == "exp":
                    # For "exp", include r0 along with other parameters
                    weighting.update({"c": c, "r0": r0})
                    d = st.number_input(
                        f"⚙️ {desc} - Weighting d", min_value=0.01, max_value=5.0, value=1.0, key=f"{desc}_weighting_d"
                    )
                    weighting["d"] = d
                    threshold = st.number_input(
                        f"⚙️ {desc} - Weighting Threshold", min_value=1e-6, max_value=1e-1, value=1e-2,
                        key=f"{desc}_weighting_threshold"
                    )
                    weighting["threshold"] = threshold

                # Optional w0 parameter for all functions
                use_w0 = st.checkbox(f"⚙️ {desc} - Use w0 (Override Central Atoms)", key=f"{desc}_weighting_use_w0")
                if use_w0:
                    w0 = st.number_input(
                        f"⚙️ {desc} - w0 Weight", min_value=0.0, max_value=5.0, value=0.0, key=f"{desc}_weighting_w0"
                    )
                    weighting["w0"] = w0

            # Averaging mode
            average = st.selectbox(f"⚙️ {desc} - Averaging Mode", ["off", "inner", "outer"], index=0,
                                   key=f"{desc}_average",
                                   help="Defines how SOAP features are averaged.")

            # Compression settings
            compression_mode = st.selectbox(f"⚙️ {desc} - Compression Mode", ["off", "mu2", "mu1nu1", "crossover"],
                                            index=0,
                                            key=f"{desc}_compression_mode",
                                            help="Defines how SOAP features are compressed.")
            compression = {"mode": compression_mode}
            if compression_mode == "mu2":
                species_weighting = st.text_input(f"⚙️ {desc} - Species Weighting (comma-separated)", "",
                                                  key=f"{desc}_species_weighting")
                if species_weighting:
                    species_weights = {s.strip(): float(w) for s, w in zip(species, species_weighting.split(","))}
                    compression["species_weighting"] = species_weights

            # Additional boolean parameters
            periodic = st.checkbox(f"⚙️ {desc} - Periodic", value=True, key=f"{desc}_periodic",
                                   help="Enable periodic boundary conditions.")

            sparse = st.checkbox(f"⚙️ {desc} - Sparse Output", value=False, key=f"{desc}_sparse",
                                 help="Output as a sparse matrix.")

            dtype = st.selectbox(f"⚙️ {desc} - Data Type", ["float32", "float64"], index=1, key=f"{desc}_dtype",
                                 help="Define the data type for SOAP output.")

            descriptor_parameters[desc] = {
                "species": species,
                "r_cut": r_cut,
                "n_max": n_max,
                "l_max": l_max,
                "sigma": sigma,
                "rbf": rbf,
                "weighting": weighting,
                "average": average,
                "compression": compression,
                "periodic": periodic,
                "sparse": sparse,
                "dtype": dtype,
            }
        elif desc == "ValleOganov":
            species_input = st.text_input(f"{desc} - Species", ", ".join(species_list), key=f"{desc}_species",
                                          help="Comma-separated list of atomic species (e.g., H, O, C).")
            species = [s.strip() for s in species_input.split(",")]
            function = st.selectbox(f"{desc} - Function", ["distance", "angle"], index=0, key=f"{desc}_function",
                                    help="Select the geometry function ('distance' for pairwise distances or 'angle' for angles).")
            sigma = st.number_input(f"{desc} - Sigma", min_value=1e-6, max_value=10.0, value=10 ** (-0.5),
                                    key=f"{desc}_sigma", help="Standard deviation of the Gaussian broadening.")
            n = st.number_input(f"{desc} - n", min_value=10, max_value=500, value=100, step=10,
                                key=f"{desc}_n", help="Number of discretization points or bins.")
            r_cut = st.number_input(f"{desc} - Cutoff Radius (Å)", min_value=1.0, max_value=10.0, value=8.0,
                                    key=f"{desc}_r_cut", help="Radial cutoff (maximum distance in Å).")
            sparse = st.checkbox(f"⚙️ {desc} - Sparse Output", value=False, key=f"{desc}_sparse",
                                 help="Return output as a sparse matrix (Default: False).")

            # New: Add dtype parameter
            dtype = st.selectbox(f"⚙️ {desc} - Data Type", ["float32", "float64"], index=1, key=f"{desc}_dtype",
                                 help="Data type for the output (Default: float64).")

            descriptor_parameters[desc] = {
                "species": species,
                "function": function,
                "sigma": sigma,
                "n": n,
                "r_cut": r_cut,
                "sparse": sparse,
                "dtype": dtype,
            }
        elif desc == "PRDF":
            cutoff = st.number_input(f"{desc} - Cutoff (Å)", min_value=5.0, max_value=50.0, value=20.0, step=1.0,
                                     format="%.1f", key=f"{desc}_cutoff")
            st.caption("Maximum distance for radial distribution (in Å).")
            bin_size = st.number_input(f"{desc} - Bin Size (Å)", min_value=0.1, max_value=5.0, value=0.2, step=0.1,
                                       format="%.1f", key=f"{desc}_bin_size")
            st.caption("Width of the bins (in Å).")
            descriptor_parameters[desc] = {"cutoff": cutoff, "bin_size": bin_size}
        else:
            st.info(f"No parameter inputs available for {desc}.")
            descriptor_parameters[desc] = {}

        try:
            structure = read(uploaded_files[0].name)
            if desc == "PRDF":
                mg_structure = AseAtomsAdaptor.get_structure(structure)
                featurizer = descriptor_options[desc](**descriptor_parameters[desc])
                featurizer.fit([mg_structure])
                descriptor_values = featurizer.featurize(mg_structure)
            else:
                featurizer = descriptor_options[desc](**descriptor_parameters.get(desc, {}))
                descriptor_values = featurizer.create(structure)
            descriptor_lengths[desc] = (len(descriptor_values.flatten())
                                        if isinstance(descriptor_values, np.ndarray)
                                        else len(descriptor_values))
        except Exception as e:
            descriptor_lengths[desc] = f"Error: {e}"



    #SDSD
    if "PRDF" in selected_descriptors:
        #cutoff = st.number_input("PRDF - Cutoff (Å)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
        #bin_size = st.number_input("PRDF - Bin Size (Å)", min_value=0.1, max_value=5.0, value=0.2, step=0.1)
        plot_prdf = st.checkbox("Plot PRDF for Each Element Combination", value=False)

        if uploaded_files and plot_prdf:
            bins = np.arange(0, cutoff + bin_size, bin_size)  # Ensure proper binning
            species_combinations = list(combinations(species_list, 2)) + [(s, s) for s in species_list]
            avg_prdf = {comb: np.zeros(len(bins) - 1) for comb in species_combinations}  # Adjusted to match bin count

            num_structures = 0

            all_prdf_dict = defaultdict(list)
            all_distance_dict = {}

            # Process multiple uploaded structures
            for uploaded_file in uploaded_files:

                structure = read(uploaded_file.name)
                mg_structure = AseAtomsAdaptor.get_structure(structure)

                # Initialize PRDF featurizer
                featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
                featurizer.fit([mg_structure])

                # Extract PRDF values
                prdf_data = featurizer.featurize(mg_structure)

                # Extract feature labels (e.g., 'Ti-Ti PRDF r=0.00-0.20')
                feature_labels = featurizer.feature_labels()

                # Temporary dictionaries for the current structure
                prdf_dict = defaultdict(list)
                distance_dict = {}

                # Extract element pairs and corresponding distances
                for i, label in enumerate(feature_labels):
                    parts = label.split(" PRDF r=")  # Split into element pair and distance
                    element_pair = tuple(parts[0].split("-"))  # Convert 'Ti-Ti' to ('Ti', 'Ti')
                    distance_range = parts[1].split("-")  # e.g., ['0.00', '0.20']

                    # Compute the bin center (midpoint of the range)
                    bin_center = (float(distance_range[0]) + float(distance_range[1])) / 2

                    # Store the PRDF values and corresponding distances
                    prdf_dict[element_pair].append(prdf_data[i])  # Ensure this is a list

                    # FIX: Store a **list** of bin centers instead of a single value
                    if element_pair not in distance_dict:
                        distance_dict[element_pair] = []  # Initialize list if not present

                    distance_dict[element_pair].append(bin_center)

                # Store PRDF values into global dictionaries for averaging
                for pair in prdf_dict:
                    if pair not in all_distance_dict:
                        all_distance_dict[pair] = distance_dict[pair]  # Save the distance bins

                    # Convert single floats to lists if needed
                    if isinstance(prdf_dict[pair], float):
                        prdf_dict[pair] = [prdf_dict[pair]]

                    all_prdf_dict[pair].append(prdf_dict[pair])  # Append list of PRDF values


            # Streamlit UI
            st.subheader("Averaged PRDF Across Structures")

            # Compute and plot averaged PRDF
            if all_prdf_dict:
                for comb, prdf_list in all_prdf_dict.items():
                    # Ensure all PRDF values are arrays
                    print(prdf_list )
                    valid_prdf = [np.array(p) for p in prdf_list if
                                  isinstance(p, list) ]
                    #valid_prdf=1
                    # Convert list of lists to NumPy array for averaging
                    if len(valid_prdf) > 0:
                        prdf_array = np.vstack(valid_prdf)  # Stack into 2D array
                        avg_prdf = np.mean(prdf_array, axis=0)  # Compute mean
                    else:
                        avg_prdf = np.zeros_like(all_distance_dict[comb])  # Avoid empty arrays
                    print(all_distance_dict[comb])
                    print(avg_prdf)
                    # Plot the averaged PRDF
                    fig, ax = plt.subplots()
                    ax.plot(all_distance_dict[comb], avg_prdf, label=f"{comb[0]}-{comb[1]}")
                    ax.set_xlabel("Distance (Å)")
                    ax.set_ylabel("PRDF Intensity")
                    ax.set_title(f"Averaged PRDF: {comb[0]}-{comb[1]}")
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.error("No valid PRDF data collected from uploaded structures.")





    st.subheader("📊 Live Descriptor Length Preview")
    formatted_preview = [f"{desc}: {descriptor_lengths[desc]}" for desc in descriptor_lengths]
    formatted_output = f"[{' '.join(descriptor_lengths.keys())}] -> [{', '.join(formatted_preview)}]"
    st.write(f"🧩 **Fingerprint Vector Length:**\n```\n{formatted_output}\n```")

    # --- Clustering Options (placed before PCA Options) ---
    st.divider()
    st.subheader("⚙️ SET: ⟶ k-Means Clustering Options")
    perform_clustering = st.checkbox("Perform Clustering", value=True)
    if perform_clustering:
        num_clusters = st.number_input("⚙️ Number of clusters (for KMeans) = Number of structures to select as the most different ones.",
                                       min_value=2, max_value=len(uploaded_files),
                                       value=5 if len(uploaded_files) >= 5 else len(uploaded_files),
                                       step=1)
        rep_method = st.selectbox("⚙️ Representative Selection Method",
                                  ["Closest to Centroid", "Farthest from Each Other"],
                                  help="Choose whether to select the structure closest to each cluster centroid or a structure from each cluster that maximizes the distance to the ones already selected (greedy approach).")

    # --- PCA Options (placed before t-SNE parameters) ---
    st.divider()
    st.subheader("⚙️ SET: ⟶ PCA Options")
    pca_option = st.checkbox("Perform PCA reduction before t-SNE", value=False)
    if pca_option:
        cum_threshold = st.slider("⚙️ Cumulative Explained Variance Threshold",
                                  min_value=0.90, max_value=1.0, value=0.98, step=0.01,
                                  format="%.2f")

    st.divider()
    st.subheader("⚙️ SET: ⟶ t-SNE Options")
    if uploaded_files:
        # Ensure at least 1 structure exists; t-SNE perplexity must be less than the number of samples.
        default_perplexity = 1 if len(uploaded_files) <= 1 else (
            len(uploaded_files) - 1 if (len(uploaded_files) - 1) < 10 else 10)
        perplexity = st.number_input("⚙️ t-SNE Perplexity", min_value=1, max_value=50, value=default_perplexity, step=1)
    #perplexity = st.number_input("t-SNE Perplexity", min_value=5, max_value=50, value=10, step=1)
    learning_rate = st.number_input("⚙️ Learning Rate", min_value=10, max_value=500, value=200, step=1)


    #run_button = st.button("Run Simulation")
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            font-size: 20px;
            padding: 15px 30px;
            background-color: #007BFF; /* Blue color */
            color: white !important; /* Force white text */
            font-weight: bold;
            border-radius: 10px;
            border: none;
            transition: background-color 0.3s ease;
        }

        div.stButton > button:first-child:hover {
            background-color: #0056b3; /* Darker blue on hover */
        }

        div.stButton > button:first-child:active {
            background-color: #004085 !important; /* Even darker blue when clicked */
            color: white !important; /* Ensure text remains white */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Now create the button normally
    run_button = st.button("Run Simulation")
else:
    run_button = False



# --- Processing and Visualization ---
if run_button:
    log_area = st.empty()
    log_messages = deque(maxlen=10)

    def log(msg):
        log_messages.append(msg)
        log_area.text("\n".join(log_messages))

    with st.spinner("Processing files..."):
        try:
            log("🚀 Starting simulation...")
            feature_matrix = []
            file_names = []
            for uploaded_file in uploaded_files:
                log(f"📂 Processing file: {uploaded_file.name}")
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                structure = read(uploaded_file.name)
                feature_vectors = []
                for desc in selected_descriptors:
                    log(f"⚙️ Computing {desc} for {uploaded_file.name}...")
                    if desc == "PRDF":
                        mg_structure = AseAtomsAdaptor.get_structure(structure)
                        featurizer = descriptor_options[desc](**descriptor_parameters[desc])
                        featurizer.fit([mg_structure])
                        descriptor_values = featurizer.featurize(mg_structure)
                    else:
                        featurizer = descriptor_options[desc](**descriptor_parameters.get(desc, {}))
                        descriptor_values = featurizer.create(structure)
                    feature_vectors.extend(
                        descriptor_values.flatten() if isinstance(descriptor_values, np.ndarray) else descriptor_values
                    )
                feature_matrix.append(feature_vectors)
                file_names.append(uploaded_file.name)
                log(f"✅ Finished file: {uploaded_file.name}")

            log("🗃️ Creating DataFrame from feature matrix...")
            df = pd.DataFrame(feature_matrix, index=file_names)

            log("🧹 Imputing missing values, if any...")
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

            # --- PCA Reduction ---
            # --- PCA Reduction ---
            if pca_option:
                log("🔎 Performing PCA reduction...")
                max_possible = min(df_imputed.shape)
                pca_temp = PCA(n_components=max_possible)
                pca_temp.fit(df_imputed)
                explained_variance_ratio = pca_temp.explained_variance_ratio_
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                if any(cumulative_explained_variance > cum_threshold):
                    n_components = np.where(cumulative_explained_variance > cum_threshold)[0][0] + 1
                else:
                    n_components = max_possible
                st.write(
                    f"Choosing {n_components} PCA components to reach a cumulative explained variance threshold of {cum_threshold:.2f}."
                )
                # Plot cumulative explained variance curve
                fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
                ax_pca.plot(np.arange(1, max_possible + 1), cumulative_explained_variance, marker='o', linestyle='-')
                ax_pca.axhline(y=cum_threshold, color='r', linestyle='--', label=f"Threshold = {cum_threshold:.2f}")
                ax_pca.set_xlabel("Number of Components")
                ax_pca.set_ylabel("Cumulative Explained Variance")
                ax_pca.set_title("Cumulative Explained Variance vs. Number of PCA Components")
                ax_pca.legend()
                st.subheader("📊 PCA Cumulative Explained Variance")
                st.pyplot(fig_pca)

                pca = PCA(n_components=n_components, whiten=False, random_state=42)
                df_pca = pca.fit_transform(df_imputed)
                st.write(f"PCA-reduced fingerprints shape: {df_pca.shape}")
                data_for_tsne = df_pca
            else:
                # Convert the imputed DataFrame to a numpy array for consistent indexing.
                data_for_tsne = df_imputed.to_numpy()  # or use df_imputed.values
            # --- End PCA Reduction ---

            # --- Clustering ---
            # --- Clustering ---
            if perform_clustering:
                log("🔎 Performing KMeans clustering on final fingerprints...")
                kmeans = KMeans(
                    n_clusters=num_clusters, random_state=0, tol=1e-6,
                    algorithm='lloyd', max_iter=600, n_init=num_clusters - round(0.5 * num_clusters)
                ).fit(data_for_tsne)
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
                selected_structures = {}
                representative_indices = {}
                if rep_method == "Closest to Centroid":
                    for i in range(num_clusters):
                        indices = np.where(labels == i)[0]
                        if len(indices) == 0:
                            selected_structures[i] = "No structure in cluster"
                            representative_indices[i] = None
                        else:
                            cluster_points = data_for_tsne[indices, :]
                            dists = np.linalg.norm(cluster_points - centroids[i], axis=1)
                            best_index = indices[np.argmin(dists)]
                            selected_structures[i] = file_names[best_index]
                            representative_indices[i] = best_index
                elif rep_method == "Farthest from Each Other":
                    # Greedy approach: select one candidate per cluster that maximizes the minimum distance to already selected candidates.
                    selected_indices = []
                    for i in range(num_clusters):
                        indices = np.where(labels == i)[0]
                        if len(indices) == 0:
                            selected_structures[i] = "No structure in cluster"
                            representative_indices[i] = None
                            continue
                        candidate_points = data_for_tsne[indices, :]
                        if len(selected_indices) == 0:
                            # For the first cluster, choose the candidate farthest from the global mean.
                            global_mean = np.mean(data_for_tsne, axis=0)
                            distances = np.linalg.norm(candidate_points - global_mean, axis=1)
                            best_index = indices[np.argmax(distances)]
                        else:
                            # For each candidate in the current cluster, compute its minimum distance to all already selected representatives.
                            selected_points = data_for_tsne[selected_indices, :]
                            distances = [np.min(np.linalg.norm(candidate - selected_points, axis=1)) for candidate in
                                         candidate_points]
                            best_index = indices[np.argmax(distances)]
                        selected_indices.append(best_index)
                        selected_structures[i] = file_names[best_index]
                        representative_indices[i] = best_index

                with st.expander("View Representative Structure Names"):
                    rep_str = "\n".join([f"Cluster {i}: {selected_structures[i]}" for i in range(num_clusters)])
                    st.text(rep_str)

                # --- New Expander: Structures Within Each Cluster ---
                with st.expander("View All Structures in Each Cluster"):
                    cluster_details = []
                    for i in range(num_clusters):
                        indices = np.where(labels == i)[0]
                        if len(indices) == 0:
                            cluster_details.append(f"Cluster {i}: No structures in cluster")
                        else:
                            cluster_files = ", ".join([file_names[idx] for idx in indices])
                            cluster_details.append(f"Cluster {i}: {cluster_files}")
                    st.text("\n".join(cluster_details))
            else:
                labels = None
            # --- End Clustering ---

            log("🔎 Running t-SNE analysis...")
            fig_tsne, ax_tsne = plt.subplots(figsize=(6, 4))
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
            tsne_results = tsne.fit_transform(data_for_tsne)
            log("🎉 t-SNE computation complete.")

            if labels is not None:
                vmin = np.min(labels)
                vmax = np.max(labels)
                scatter = ax_tsne.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7,
                                          vmin=vmin, vmax=vmax)

                # Mark representative structures with black empty circles.
                for i in range(num_clusters):
                    rep_index = representative_indices.get(i)
                    if rep_index is not None:
                        ax_tsne.scatter(tsne_results[rep_index, 0], tsne_results[rep_index, 1], s=150,
                                        facecolors='none', edgecolors='black', linewidths=2)
            else:
                ax_tsne.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)

            ax_tsne.set_title("t-SNE Projection of Uploaded Structures")
            ax_tsne.set_xlabel("t-SNE Component 1")
            ax_tsne.set_ylabel("t-SNE Component 2")
            st.pyplot(fig_tsne)
            # --- t-SNE in 3D Space with Representative Markers ---
            log("🔎 Running 3D t-SNE analysis...")
            tsne_3d = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
            tsne_results_3d = tsne_3d.fit_transform(data_for_tsne)

            from mpl_toolkits.mplot3d import Axes3D
            fig_tsne_3d = plt.figure(figsize=(8, 6))
            ax_tsne_3d = fig_tsne_3d.add_subplot(111, projection='3d')

            if labels is not None:
                vmin = np.min(labels)
                vmax = np.max(labels)
                scatter3d = ax_tsne_3d.scatter(tsne_results_3d[:, 0],
                                               tsne_results_3d[:, 1],
                                               tsne_results_3d[:, 2],
                                               c=labels,
                                               cmap='tab10',
                                               alpha=0.7,
                                               vmin=vmin,
                                               vmax=vmax)

                for i in range(num_clusters):
                    rep_index = representative_indices.get(i)
                    if rep_index is not None:
                        ax_tsne_3d.scatter(tsne_results_3d[rep_index, 0],
                                           tsne_results_3d[rep_index, 1],
                                           tsne_results_3d[rep_index, 2],
                                           s=150, facecolors='none', edgecolors='black', linewidths=2)
            else:
                ax_tsne_3d.scatter(tsne_results_3d[:, 0],
                                   tsne_results_3d[:, 1],
                                   tsne_results_3d[:, 2],
                                   alpha=0.7)

            ax_tsne_3d.set_title("3D t-SNE Projection of Uploaded Structures")
            ax_tsne_3d.set_xlabel("t-SNE Component 1")
            ax_tsne_3d.set_ylabel("t-SNE Component 2")
            ax_tsne_3d.set_zlabel("t-SNE Component 3")
            st.pyplot(fig_tsne_3d)

            log("📍 Displaying structures with identical t-SNE positions...")
            tsne_positions = defaultdict(list)
            for i, position in enumerate(tsne_results):
                pos_tuple = tuple(position)
                tsne_positions[pos_tuple].append(file_names[i])
            st.subheader("📊 Structures with Identical t-SNE Positions:")
            for pos, structures in tsne_positions.items():
                if len(structures) > 1:
                    st.write(f"📌 Position `{pos}`: **{', '.join(structures)}**")
            log("🏁 Simulation complete.")

        except Exception as e:
            st.error(f"❌ Error processing files: {e}")
            log(f"🚫 Error encountered: {e}")
