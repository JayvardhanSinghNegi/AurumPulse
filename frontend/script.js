async function submitSequence() {
  const input = document.getElementById("price-sequence").value.trim();
  const sequence = input.split(',').map(Number).filter(val => !isNaN(val));
  const userType = document.getElementById("user-type").value;
  const resultDiv = document.getElementById("prediction-result");
  const emotionDiv = document.getElementById("emotion-result");
  const submitBtn = document.getElementById("submit-button");

  resultDiv.innerText = "";
  emotionDiv.innerText = "";

  if (sequence.length !== 30) {
    alert("Please enter exactly 30 valid price values, separated by commas.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.innerText = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        sequence: sequence,
        user_type: userType
      })
    });

    const data = await response.json();

    if (response.ok && data.predicted_price !== undefined) {
      resultDiv.innerText = "Predicted Price: $" + data.predicted_price;
      emotionDiv.innerText = data.sentiment || "";
    } else {
      resultDiv.innerText = "Error: " + (data.error || "Unknown error");
    }
  } catch (error) {
    resultDiv.innerText = "Failed to connect to backend.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerText = "Predict";
  }
}
