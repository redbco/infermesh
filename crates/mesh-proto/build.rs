use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Configure tonic-build 0.12 - re-enable client generation
    let config = tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir);

    // Compile the proto files
    config.compile_protos(
        &[
            "proto/control_plane.proto",
            "proto/state_plane.proto", 
            "proto/scoring.proto",
        ],
        &["proto"],
    )?;

    // Tell cargo to rerun this build script if the proto files change
    println!("cargo:rerun-if-changed=proto/control_plane.proto");
    println!("cargo:rerun-if-changed=proto/state_plane.proto");
    println!("cargo:rerun-if-changed=proto/scoring.proto");
    
    Ok(())
}