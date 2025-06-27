#!/usr/bin/env python3
"""
PoUW Account Export Tool

This script helps you export your PoUW account data to transfer to another device.
It creates a backup of your blockchain database and node configurations.
"""

import os
import shutil
import json
import tarfile
import datetime
from pathlib import Path

def export_account(node_id="cpu-miner", export_dir="./account_backup"):
    """Export PoUW account data for transfer to another device"""
    
    print("🔄 PoUW Account Export Tool")
    print("=" * 40)
    
    # Create export directory
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    
    # Files to export
    files_to_export = {
        "blockchain.db": "Blockchain database (contains your coins)",
        f"configs/{node_id}.json": f"Node configuration for {node_id}",
    }
    
    # Check what files exist
    existing_files = {}
    missing_files = {}
    
    for file_path, description in files_to_export.items():
        if os.path.exists(file_path):
            existing_files[file_path] = description
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({file_size:,} bytes) - {description}")
        else:
            missing_files[file_path] = description
            print(f"❌ {file_path} - {description} (NOT FOUND)")
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files are missing!")
        print("The export will continue with available files only.")
    
    # Copy files to export directory
    print(f"\n📦 Copying files to {export_dir}/...")
    
    copied_files = []
    for file_path in existing_files.keys():
        dest_path = export_path / os.path.basename(file_path)
        try:
            shutil.copy2(file_path, dest_path)
            copied_files.append(file_path)
            print(f"✅ Copied {file_path}")
        except Exception as e:
            print(f"❌ Failed to copy {file_path}: {e}")
    
    # Create account info file
    account_info = {
        "node_id": node_id,
        "miner_address": f"miner_{node_id}",
        "export_date": datetime.datetime.now().isoformat(),
        "files_exported": copied_files,
        "instructions": {
            "1": "Copy the entire account_backup folder to your new device",
            "2": "Install PoUW on the new device",
            "3": "Run the import_account.py script on the new device",
            "4": "Your coins and mining history will be available"
        }
    }
    
    with open(export_path / "account_info.json", 'w') as f:
        json.dump(account_info, f, indent=2)
    
    # Create compressed archive
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"pouw_account_{node_id}_{timestamp}.tar.gz"
    
    print(f"\n📦 Creating compressed archive: {archive_name}")
    
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(export_dir, arcname=f"pouw_account_{node_id}")
    
    archive_size = os.path.getsize(archive_name)
    print(f"✅ Archive created: {archive_name} ({archive_size:,} bytes)")
    
    # Show summary
    print(f"\n🎉 Export completed successfully!")
    print("=" * 40)
    print(f"📁 Export folder: {export_dir}")
    print(f"📦 Archive file: {archive_name}")
    print(f"💰 Node ID: {node_id}")
    print(f"🏦 Miner address: miner_{node_id}")
    print()
    print("📋 Next steps:")
    print("1. Transfer the .tar.gz file to your new device")
    print("2. Extract it: tar -xzf " + archive_name)
    print("3. Install PoUW on the new device")
    print("4. Run: python3 import_account.py")
    print()
    
    return archive_name

if __name__ == "__main__":
    import sys
    
    # Get node ID from command line or use default
    node_id = sys.argv[1] if len(sys.argv) > 1 else "cpu-miner"
    
    try:
        archive_name = export_account(node_id)
        print(f"✅ Account export successful: {archive_name}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1) 