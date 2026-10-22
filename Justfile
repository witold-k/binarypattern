# user_name        := env("USER")
# current_location := justfile()
current_dir      := justfile_directory()
module_name      := file_name(current_dir)

default: build

build:
    cargo build
    RUST_BACKTRACE=1 cargo test
    cargo clippy

build-jvm:
    cargo build --target wasm32-unknown-unknown --release

fix:
    aifix -t fix_code -f {{current_dir}} -f {{current_dir}}/..

clean:
	@cargo clean -p {{module_name}}

clean-all:
	@rm target -rf
	@cargo clean


