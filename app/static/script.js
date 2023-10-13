
let dummyData;

// Load dummy data from JSON
fetch('./static/testdata.json')
    .then(response => response.json())
    .then(data => {
        dummyData = data;
        console.log('Dummy data loaded:', dummyData);
        populateDropdown();
    })
    .catch(error => console.error('Error loading dummy data:', error));

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
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems);

    console.log('Dropdown populated with options:', dropdown.innerHTML);
}



function fillFormWithDummyDataAndPredict() {
    const selectedOption = document.getElementById('dummyDataDropdown').value;
    const data = dummyData[selectedOption];

    if (data) {
        // Fill the form fields with dummy data
        document.getElementById('first_name').value = data.applicantData.first_name;
        document.getElementById('last_name').value = data.applicantData.last_name;
        document.getElementById('age').value = data.applicantData.age;
        document.getElementById('race').value = data.applicantData.race;
        document.getElementById('sex').value = data.applicantData.sex;
        document.getElementById('ethnicity').value = data.applicantData.ethnicity;
        document.getElementById('income').value = data.applicantData.income;
        document.getElementById('debtToIncomeRatio').value = data.applicantData.debtToIncomeRatio;
        document.getElementById('creditScore').value = data.applicantData.creditScore;
        document.getElementById('lenderCredits').value = data.applicantData.lenderCredits;

        document.getElementById('applicantAge').value = data.coApplicantData.applicantAge;
        document.getElementById('applicantRace').value = data.coApplicantData.applicantRace;
        document.getElementById('applicantSex').value = data.coApplicantData.applicantSex;
        document.getElementById('applicantEthnicity').value = data.coApplicantData.applicantEthnicity;

        document.getElementById('propertyvalue').value = data.propertyInfo.propertyValue;
        document.getElementById('lienstatus').value = data.propertyInfo.lienStatus;
        document.getElementById('manufacturedHomeLandPropertyInterest').value = data.propertyInfo.manufacturedHomeLandPropertyInterest;
        document.getElementById('multifamilyAffordableUnits').value = data.propertyInfo.multifamilyAffordableUnits;
        document.getElementById('occupancyType').value = data.propertyInfo.occupancyType;
        document.getElementById('manufacturedHomeSecuredPropertyType').value = data.propertyInfo.manufacturedHomeSecuredPropertyType;
        document.getElementById('totalUnits').value = data.propertyInfo.totalUnits;

        document.getElementById('loanAmount').value = data.loanDetails.loanAmount;
        document.getElementById('interestRate').value = data.loanDetails.interestRate;
        document.getElementById('totalpointsandfees').value = data.loanDetails.totalpointsandfees;
        document.getElementById('loanterm').value = data.loanDetails.loanterm;
        document.getElementById('discountPoints').value = data.loanDetails.discountPoints;
        document.getElementById('prepaymentPenaltyTerm').value = data.loanDetails.prepaymentPenaltyTerm;
        document.getElementById('negativeAmortization').value = data.loanDetails.negativeAmortization;
        document.getElementById('totalloancosts').value = data.loanDetails.totalloancosts;
        document.getElementById('loantype').value = data.loanDetails.loantype;
        document.getElementById('loanpurpose').value = data.loanDetails.loanpurpose;
        document.getElementById('originationCharges').value = data.loanDetails.originationCharges;
        document.getElementById('interestOnlyPayment').value = data.loanDetails.interestOnlyPayment;
        document.getElementById('balloonPayment').value = data.loanDetails.balloonPayment;
        document.getElementById('otherNonamortizingFeatures').value = data.loanDetails.otherNonamortizingFeatures;
    } else {
        console.error('No data found for the selected option.');
    }
}

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
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(jsonData)
    })
        .then(response => response.json())
        .then(data => {
            console.log('Prediction:', data);
            // Update your UI with the prediction
            document.getElementById('prediction').innerText = `Predicted class: ${data.prediction}`;
        })
        .catch(error => console.error('Error predicting:', error));
}

// Attach this function to the form submission
document.getElementById('loanForm').addEventListener('submit', function (event) {
    event.preventDefault();
    predictLoan(); // Call the function to predict the loan
});




// Initialize Materialize select elements
document.addEventListener('DOMContentLoaded', function () {
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems);
});

document.querySelector('form').addEventListener('submit', function (event) {
    event.preventDefault();

    // Gather all the form values
    const age = document.getElementById('age').value;
    const income = document.getElementById('income').value;
    const race = document.getElementById('race').value;
    const sex = document.getElementById('sex').value;
    const ethnicity = document.getElementById('ethnicity').value;
    const lenderCredits = document.getElementById('lenderCredits').value;
    const debtToIncomeRatio = document.getElementById('debtToIncomeRatio').value;
    const creditScore = document.getElementById('creditScore').value;

    const loanAmount = document.getElementById('loanAmount').value;
    const interestRate = document.getElementById('interestRate').value;
    const totalpointsandfees = document.getElementById('totalpointsandfees').value;
    const loanterm = document.getElementById('loanterm').value;
    const discountPoints = document.getElementById('discountPoints').value;
    const prepaymentPenaltyTerm = document.getElementById('prepaymentPenaltyTerm').value;
    const negativeAmortization = document.getElementById('negativeAmortization').value;
    const totalloancosts = document.getElementById('totalloancosts').value;
    const loantype = document.getElementById('loantype').value;
    const loanpurpose = document.getElementById('loanpurpose').value;
    const originationCharges = document.getElementById('originationCharges').value;
    const interestOnlyPayment = document.getElementById('interestOnlyPayment').value;
    const balloonPayment = document.getElementById('balloonPayment').value;
    const otherNonamortizingFeatures = document.getElementById('otherNonamortizingFeatures').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'age': age,
            'income': income,
            'race': race,
            'sex': sex,
            'ethnicity': ethnicity,
            'lenderCredits': lenderCredits,
            'debtToIncomeRatio': debtToIncomeRatio,
            'creditScore': creditScore,

            'loanAmount': loanAmount,
            'interestRate': interestRate,
            'totalpointsandfees': totalpointsandfees,
            'loanterm': loanterm,
            'discountPoints': discountPoints,
            'prepaymentPenaltyTerm': prepaymentPenaltyTerm,
            'negativeAmortization': negativeAmortization,
            'totalloancosts': totalloancosts,
            'loantype': loantype,
            'loanpurpose': loanpurpose,
            'originationCharges': originationCharges,
            'interestOnlyPayment': interestOnlyPayment,
            'balloonPayment': balloonPayment,
            'otherNonamortizingFeatures': otherNonamortizingFeatures,
        })
    })
        .then(response => response.json())
        .then(data => {
            const predictionElement = document.getElementById('prediction');
            predictionElement.innerText = `Predicted class: ${data.prediction}`;
        });
});
