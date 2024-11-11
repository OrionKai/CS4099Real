fn main() {
    println!("Hello, Rust!");

    // Simple arithmetic operations
    let a = 10;
    let b = 20;
    let sum = a + b;
    let product = a * b;

    println!("a = {}, b = {}", a, b);
    println!("Sum: {}", sum);
    println!("Product: {}", product);

    // Conditional statement
    if sum > product {
        println!("Sum is greater than product.");
    } else {
        println!("Sum is not greater than product.");
    }

    // Loop example
    for i in 1..=5 {
        println!("Counting: {}", i);
    }

    // Using a function
    let result = multiply(a, b);
    println!("Result from multiply function: {}", result);
}

// A simple function to multiply two numbers
fn multiply(x: i32, y: i32) -> i32 {
    x * y
}
