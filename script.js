const body = document.querySelector('body'),
      sidebar = body.querySelector('nav'),
      toggle = body.querySelector(".toggle"),
      searchBtn = body.querySelector(".search-box"),
      modeSwitch = body.querySelector(".toggle-switch"),
      modeText = body.querySelector(".mode-text");


toggle.addEventListener("click" , () =>{
    sidebar.classList.toggle("close");
})

searchBtn.addEventListener("click" , () =>{
    sidebar.classList.remove("close");
})

modeSwitch.addEventListener("click" , () =>{
    body.classList.toggle("dark");
    
    if(body.classList.contains("dark")){
        modeText.innerText = "Light mode";
    }else{
        modeText.innerText = "Dark mode";
        
    }
});

function submitPayment() {
    const cardNumber = document.getElementById('cardNumber').value;
    const expiryDate = document.getElementById('expiryDate').value;
    const cvv = document.getElementById('cvv').value;

    // Add your payment processing logic here (this is just a placeholder)
    const paymentResult = simulatePayment(cardNumber, expiryDate, cvv);

    // Display the payment result
    const paymentResultElement = document.getElementById('paymentResult');
    paymentResultElement.textContent = paymentResult;
}

function simulatePayment(cardNumber, expiryDate, cvv) {
    // Placeholder for payment processing logic
    // In a real-world scenario, you would send this data to a server for processing
    // and handle the response accordingly.

    // For demonstration purposes, this is a simple simulation
    if (cardNumber && expiryDate && cvv) {
        return 'Payment successful!';
    } else {
        return 'Payment failed. Please check your details.';
    }
}
