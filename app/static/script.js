// Event Listeners
document.addEventListener('DOMContentLoaded', function () {
    // Materialize select elements
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems);
});

document.getElementById('loanForm').addEventListener('submit', function (event) {
    event.preventDefault();
    handleFormSubmission();
});

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

// Gather Form Data
function gatherFormData() {
    // Gather all the form values
    const formData = {};
    const formElements = [
        'age', 'income', 'race', 'sex', 'ethnicity', 'lenderCredits',
        'debtToIncomeRatio', 'creditScore', 'loanAmount', 'interestRate',
        'totalpointsandfees', 'loanterm', 'discountPoints', 'prepaymentPenaltyTerm',
        'negativeAmortization', 'totalloancosts', 'loantype', 'loanpurpose',
        'originationCharges', 'interestOnlyPayment', 'balloonPayment',
        'otherNonamortizingFeatures'
    ];

    formElements.forEach(element => {
        formData[element] = document.getElementById(element).value;
    });

    return formData;
}

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

    console.log('Fill with dummy data', data);

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
        console.log('Dummy data loaded:', dummyData);
        populateDropdown();
    })
    .catch(error => console.error('Error loading dummy data:', error));


/* ----------------------Predict loan acceptance from form data ---------------------------   */

function predictLoan() {
    // Prepare the data to send to the Flask app
    const formData = new FormData(document.getElementById('loanForm'));
    const jsonData = {};

    formData.forEach((value, key) => {
        jsonData[key] = value;
    });

    // Log the data being sent
    console.log('Data being sent:', jsonData);

    // Send the data to the Flask app for prediction
    const predictionEndpoint = '/predict';

    const requestOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(jsonData)
    };

    fetch(predictionEndpoint, requestOptions)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Prediction:', data);
            // Update the UI with the prediction
            const predictionElement = document.getElementById('prediction');
            predictionElement.innerText = `Predicted class: ${data.prediction}`;
        })
        .catch(error => console.error('Error predicting:', error));
}
