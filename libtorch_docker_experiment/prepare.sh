# Compiles the Rust source code, moves the compiled code to the script folder
# and prepares the Docker container
cargo build --release
cp target/release/wasmedge-wasinn-example-mobilenet-image script/
docker buildx build -t sample:v1 .
docker run -it sample:v1