## False Data Injection Attack (FDIA) in Smart Grid Systems

Smart power grids are an example of a complex cyber-physical system that relies on exchanging measurement data among entities for situational awareness and to make operational choices. Ensuring accurate and reliable data is transferred through the system is critical for making correct operational decisions. Unfortunately, false data injection attacks (FDIAs) pose a significant threat to the integrity of this data. Hostile actors can manipulate measurement information, like tampering with sensor readings, leading to decisions that could result in incorrect system state estimations, operational inefficiencies, or even catastrophic failures. 


Detecting False Data Injection Attacks (FDIAs) in power grid systems is crucial, and using machine learning techniques has become increasingly prominent in addressing this threat. Specifically, researchers have focused on leveraging machine learning strategies, such as using various neural networks for detection and mitigation efforts. These machine learning-based approaches have demonstrated significant effectiveness in identifying FDIAs. Particularly exploring the implementation of different neural network architectures, such as feed-forward neural networks (FNN), convolutional neural networks (CNN), and graph convolution networks (GCN), has yielded promising detection rates. However, ensuring the reliability and resilience of neural network-based detection mechanisms against sophisticated adversarial attacks remains an ongoing area of investigation.

## Approach
Our project aims to enhance the security of power grid systems by ensuring the robustness of binary classification systems in detecting False Data Injection Attacks (FDIAs). To achieve this, we will use the Neural Network Verification (NNV) tool and both leverage and augment its capabilities. Our approach involves two main strategies: firstly, we will rigorously verify the robustness of existing FDIA detection algorithms, and secondly, we will extend the capabilities of the NNV tool to address the unique challenges associated with power grid system security. By taking this comprehensive approach, our ultimate goal is to improve the reliability and trustworthiness of FDIA detection mechanisms, thus strengthening the power grid infrastructure against malicious cyber-attacks.

## Verification Specification
We desire trained neural networks that exhibit resilience to evolving adversarial threats. To measure this objective, we concentrate on a verification property that ensures the system's resilience, whether or not the classification accurately identifies samples even when they are subjected to modifications, such as adversarial attacks.

## Datasets

To rigorously train and evaluate, we must procure datasets that accurately reflect the operational dynamics of smart grid systems. A valuable resource is provided via a Kaggle notebook, accessible at [SMART GRID MONITORING POWER]{https://www.kaggle.com/code/pythonafroz/smart-grid-false-data-injection-attack-prediction/notebook}. This notebook offers datasets detailing both normal operations and operations affected by FDIAs within a smart grid system.

Furthermore, to collect our necessary data, we can adopt a strategy inspired by prior research involving the manual perturbation of data derived from various IEEE bus systems, available at [IEEE bus systems]{https://labs.ece.uw.edu/pstca/}. The perturbed data will serve as a foundation for training, validating, and testing our models, specifically emphasizing general attack-based FDIAs.

