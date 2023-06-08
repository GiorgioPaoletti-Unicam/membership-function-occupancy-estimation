# Optimal Membership Function Selection for Indoor Occupancy Estimation Using Unlabeled Environmental Data

This repository contains the code and resources for my Master's Thesis project focused on the integration of AI into building automation and energy management systems. The research is done in collaboration with Microsoft Switzerland 🤝 and emphasizes the development of non-intrusive, privacy-preserving solutions for indoor occupancy estimation using fuzzy logic systems.

## 📜 Abstract

This project involves the optimal selection of membership functions for estimating indoor occupancy using non-sensitive, unlabeled environmental data like temperature, humidity, light level, CO2 levels, and humidity ratio. The goal is to find an accurate indoor occupancy estimation method without compromising user privacy.

## ⚙️ Methodology

The approach involves several steps:

1. Analyzing various common membership function types and their suitability for occupancy estimation.
2. Implementing an unsupervised clustering algorithm to gain insights into the structure and trends within the unlabeled data.
3. Developing a fuzzy rule base for the fuzzy logic controller using the insights obtained from the unsupervised clustering algorithm.
4. Evaluating the performance of each membership function type for indoor occupancy estimation without compromising user privacy.
5. Identifying the optimal membership function type and its implications for building automation and energy management systems.

## 📁 Repository Structure

- `data_preprocessing`: Scripts for data preprocessing
- `membership_functions_analysis`: Scripts and notebooks for the analysis of various membership functions
- `clustering_algorithm`: Code implementing the unsupervised clustering algorithm
- `fuzzy_rule_base`: Code for developing the fuzzy rule base
- `fuzzy_logic_controller`: Implementation of the fuzzy logic controller
- `results_discussion`: Code, plots, and notebooks used in the results and discussion part of the thesis
- `docs`: Additional documentation related to the project

## 🗓️ Project Timeline

The project is planned to be completed over 5 months, with each work package and its associated tasks assigned specific durations. For more detailed information, check the `project_plan.pdf` file in the `docs` folder.

## 🧑🏻‍💻 Author
- Student: Giorgio Paoletti

## 🎓 University's
- 🇨🇭 FHNW Supervisors: Knut Hinkelmann, Emanuele Laurenzi
- 🇮🇹 UNICAM Supervisors: Andrea Polini

## 🏢 Microsoft Switzerland
- External Collaborator: Claudio Mirti

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

This project is part of the Master's Thesis for the Autumn 2023 session at UNICAM and FHNW. Graduation Day is set for 25th October 2023. 🎉
