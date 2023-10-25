

// Handle Form Submission
async function handleFormSubmission() {
    const formData = gatherFormData();
    try {
        const predictionData = await fetchPredictionData(formData);
        updatePrediction(predictionData);
    } catch (error) {
        console.error('Error predicting:', error);
    }
}
// Event Listeners
document.addEventListener('DOMContentLoaded', function () {
    // Materialize select elements
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems);

    // Add an event listener to the form submit button
    document.getElementById('loanForm').addEventListener('submit', function (event) {
        event.preventDefault();
        predictLoan();
    });
});

/* ----------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------- */

// Function to gather form data
function gatherFormData() {
    // Gather all the form values
    const formData = {};
    const formElements = [
        'age', 'income', 'race', 'sex', 'ethnicity', 'lenderCredits',
        'debtToIncomeRatio', 'creditScore', 'loanAmount', 'interestRate',
        'totalpointsandfees', 'loanterm', 'discountPoints', 'prepaymentPenaltyTerm',
        'negativeAmortization', 'totalloancosts', 'loantype', 'loanpurpose',
        'originationCharges', 'interestOnlyPayment', 'balloonPayment',
        'otherNonamortizingFeatures',
        'CoapplicantAge', 'CoapplicantRace', 'CoapplicantSex', 'CoapplicantEthnicity',
        'propertyValue', 'lienStatus', 'manufacturedHomeLandPropertyInterest',
        'multifamilyAffordableUnits', 'occupancyType', 'manufacturedHomeSecuredPropertyType',
        'totalUnits','race2','CoapplicantRace2'
    ];

    const dataMappings = {
        'age': 'applicant_age',
        'sex': 'applicant_sex',
        'race': 'applicant_race_1',
        'ethnicity': 'applicant_ethnicity_1',
        'loantype': 'loan_type',
        'income': 'income',
        'lenderCredits': 'lenderCredits',
        'debtToIncomeRatio': 'debt_to_income_ratio',
        'creditScoreType': 'applicant_credit_score_type',
        'loanAmount': 'loanAmount',
        'interestRate': 'interestRate',
        'totalpointsandfees': 'totalpointsandfees',
        'loanterm': 'loanterm',
        'discountPoints': 'discountPoints',
        'prepaymentPenaltyTerm': 'prepaymentPenaltyTerm',
        'negativeAmortization': 'negativeAmortization',
        'totalloancosts': 'totalloancosts',
        'loantype': 'loan_type',
        'loanpurpose': 'loanpurpose',
        'originationCharges': 'originationCharges',
        'interestOnlyPayment': 'interestOnlyPayment',
        'balloonPayment': 'balloonPayment',
        'otherNonamortizingFeatures': 'otherNonamortizingFeatures',
        'CoapplicantAge': 'co_applicant_age',
        'CoapplicantRace': 'co_applicant_race_1',
        'CoapplicantSex': 'co_applicant_sex',
        'CoapplicantEthnicity': 'co_applicant_ethnicity_1',
        'propertyValue': 'propertyValue',
        'lienStatus': 'lien_status',
        'manufacturedHomeLandPropertyInterest': 'manufactured_home_land_property_interest',
        'multifamilyAffordableUnits': 'multifamily_affordable_units',
        'occupancyType': 'occupancy_type',
        'manufacturedHomeSecuredPropertyType': 'manufactured_home_secured_property_type',
        'totalUnits': 'total_units',
        'race2' :'applicant_race_2',
        'CoapplicantRace2':'co_applicant_race_2',
    };

    const intDtypes = [
        'loan_type', 'lien_status', 'loan_amount', 'loan_term', 'negative_amortization',
        'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features',
        'occupancy_type', 'manufactured_home_secured_property_type',
        'manufactured_home_land_property_interest', 'total_units',
        'debt_to_income_ratio', 'applicant_age', 'co_applicant_age', 'aus_1', 'applicant_credit_score_type'
    ];

    const floatDtypes = [
        'census_tract', 'interest_rate', 'property_value', 'income',
        'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1',
        'co_applicant_race_2', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1'
    ];
    

    formElements.forEach(element => {
        const inputElement = document.getElementById(element);
        if (inputElement) {
            const mappedName = dataMappings[element] || element; // Use the mapped name or the original if no mapping
            let value = inputElement.value;

            // Convert to the appropriate data type
            if (intDtypes.includes(mappedName)) {
                value = parseInt(value);
            } else if (floatDtypes.includes(mappedName)) {
                value = parseFloat(value);
            }

            formData[mappedName] = value;
        }
    });

    console.log(formData);

    return formData;
}

/* ----------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------- */


// Fetch Prediction Data
async function fetchPredictionData(formData) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(formData)
    });
    return await response.json();
}

// Update Prediction
function updatePrediction(data) {
    const predictionElement = document.getElementById('prediction');
    predictionElement.innerText = `Predicted class: ${data.prediction}`;
}


/* ----------------------Populate form with dummy data----------------------------   */
// Populate dropdown with dummy data
let dummyData;

function populateDropdown() {
    const dropdown = document.getElementById('dummyDataDropdown');
    for (const key in dummyData) {
        if (dummyData.hasOwnProperty(key)) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = key;
            dropdown.appendChild(option);
        }
    }

    // Initialize Materialize select elements
    const elems = document.querySelectorAll('select');
    M.FormSelect.init(elems);

    console.log('Dropdown populated with options:', dropdown.innerHTML);
}

// Fill form with dummy data based on the selected option
function fillFormWithDummyData() {
    const selectedOption = document.getElementById('dummyDataDropdown').value;
    const data = dummyData[selectedOption];

    //console.log('Fill with dummy data', data);

    if (data) {
        // Iterate through the data object and its nested objects
        for (const category in data) {
            const categoryData = data[category];
            for (const field in categoryData) {
                const fieldValue = categoryData[field];
                const element = document.getElementById(field);
                if (element) {
                    element.value = fieldValue;
                }
            }
        }
    } else {
        console.error('No data found for the selected option.');
    }
}


// Load dummy data from JSON
fetch('./static/testdata.json')
    .then(response => response.json())
    .then(data => {
        dummyData = data;
        //console.log('Dummy data loaded:', dummyData);
        populateDropdown();
    })
    .catch(error => console.error('Error loading dummy data:', error));


/* ----------------------Predict loan acceptance from form data ---------------------------   */
// Function to send form data for prediction
async function predictLoan() {
    // Define the data to be sent
    const formData = gatherFormData(); // Assuming gatherFormData function is defined

    // Make the AJAX request
    $.ajax({
        type: 'POST',
        contentType: 'application/json',
        url: '/predict', 
        dataType: 'json',
        data: JSON.stringify(formData), // Sending the formData
        success: function (result) {
            console.log('Received result:', result);
            const predictionElement = document.getElementById('prediction');
            predictionElement.innerText = `Predicted class: ${result.prediction}`;
        },
        error: function (error) {
            console.error('Error:', error);
            const predictionElement = document.getElementById('prediction');
            predictionElement.innerText = `error: ${error.message}`;
        }
    });
}