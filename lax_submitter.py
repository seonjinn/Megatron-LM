import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str, help="The script to submit")
    parser.add_argument("--dry-run", action="store_true", help="Dry run the script")
    args = parser.parse_args()

    with open(args.script, "r") as f:
        content = f.read()

    content = content.replace("llmservice_fm_vision", "llmservice_nemotron_super")
    content = content.replace("batch_block1,batch_large,batch_long", "batch,batch_large,batch_large_long,batch_long")
    content = content.replace("/lustre", "/scratch")

    script = args.script.replace(".sh", "")

    new_script = f"{script}_lax.sh"
    with open(new_script, "w") as f:
        f.write(content)

    if not args.dry_run:
        subprocess.run(["sbatch", new_script])
