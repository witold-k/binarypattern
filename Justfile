user_name        := env("USER")
current_location := justfile()
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

cover:
	CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' LLVM_PROFILE_FILE='target/coverage/cargo-test-%p-%m.profraw' cargo test
	grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o target/coverage/html
	firefox target/coverage/html/index.html

keygen:
	gpg --full-generate-key

keyexport:
	gpg --armor --export -a Witold > gpg.pub
	gpg --armor --export-secret-keys -a Witold > gpg.priv

keyimport:
	gpg --import gpg.pub
	gpg --import gpg.priv

# github.com/spwhitton/git-remote-gcrypt
addsecret:
	git remote add cryptremote gcrypt::rsync://vsrv/home/{{user_name}}/depots/{{module_name}}

removesecret:
	git remote remove cryptremote

pushsecret:
	git push cryptremote master

pullsecret:
	git pull cryptremote master

helpsecret:
    gpg --list-keys

initpullsecret: helpsecret removesecret addsecret pushsecret

