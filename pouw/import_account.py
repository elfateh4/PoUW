#!/usr/bin/env python3
"""
PoUW Account Import Tool

This script helps you import your PoUW account data on a new device.
It restores your blockchain database and node configurations.
"""

import os
import shutil
import json
import tarfile
from pathlib import Path

def import_account(archive_path=None, extract_dir="./imported_account"):
    """Import PoUW account data from exported archive"""
    
    print("📥 PoUW Account Import Tool")
    print("=" * 40)
    
    # Find archive file if not specified
    if not archive_path:
        # Look for .tar.gz files in current directory
        tar_files = [f for f in os.listdir('.') if f.startswith('pouw_account_') and f.endswith('.tar.gz')]
        
        if not tar_files:
            print("❌ No PoUW account archive found!")
            print("Please specify the archive file path or place it in current directory")
            return False
        elif len(tar_files) == 1:
            archive_path = tar_files[0]
            print(f"📦 Found archive: {archive_path}")
        else:
            print("📦 Multiple archives found:")
            for i, f in enumerate(tar_files, 1):
                print(f"  {i}. {f}")
            choice = input("Select archive number: ").strip()
            try:
                archive_path = tar_files[int(choice) - 1]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return False
    
    if not os.path.exists(archive_path):
        print(f"❌ Archive file not found: {archive_path}")
        return False
    
    # Extract archive
    print(f"\n📦 Extracting {archive_path}...")
    
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(".")
            
        # Find extracted directory
        extracted_dirs = [f for f in os.listdir('.') if f.startswith('pouw_account_') and os.path.isdir(f)]
        if not extracted_dirs:
            print("❌ No account directory found in archive")
            return False
            
        account_dir = extracted_dirs[0]
        print(f"✅ Extracted to: {account_dir}")
        
    except Exception as e:
        print(f"❌ Failed to extract archive: {e}")
        return False
    
    # Read account info
    account_info_path = os.path.join(account_dir, "account_info.json")
    if os.path.exists(account_info_path):
        with open(account_info_path, 'r') as f:
            account_info = json.load(f)
        
        print(f"\n📋 Account Information:")
        print(f"  Node ID: {account_info['node_id']}")
        print(f"  Miner Address: {account_info['miner_address']}")
        print(f"  Export Date: {account_info['export_date']}")
        print(f"  Files: {len(account_info['files_exported'])}")
    else:
        print("⚠️  No account info found, proceeding with basic import")
        account_info = {"node_id": "cpu-miner"}
    
    # Check current directory structure
    print(f"\n🔍 Checking current PoUW installation...")
    
    # Create necessary directories
    os.makedirs("configs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("pids", exist_ok=True)
    os.makedirs("keys", exist_ok=True)
    
    # Import files
    files_imported = []
    
    # Import blockchain database
    blockchain_src = os.path.join(account_dir, "blockchain.db")
    if os.path.exists(blockchain_src):
        if os.path.exists("blockchain.db"):
            backup_name = f"blockchain_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.move("blockchain.db", backup_name)
            print(f"📁 Backed up existing blockchain to: {backup_name}")
        
        shutil.copy2(blockchain_src, "blockchain.db")
        db_size = os.path.getsize("blockchain.db")
        print(f"✅ Imported blockchain database ({db_size:,} bytes)")
        files_imported.append("blockchain.db")
    else:
        print("❌ No blockchain database found in archive")
    
    # Import configuration
    node_id = account_info['node_id']
    config_src = os.path.join(account_dir, f"{node_id}.json")
    config_dest = f"configs/{node_id}.json"
    
    if os.path.exists(config_src):
        if os.path.exists(config_dest):
            backup_name = f"configs/{node_id}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.move(config_dest, backup_name)
            print(f"📁 Backed up existing config to: {backup_name}")
        
        shutil.copy2(config_src, config_dest)
        print(f"✅ Imported configuration: {config_dest}")
        files_imported.append(config_dest)
    else:
        print(f"❌ No configuration found for {node_id}")
    
    # Cleanup extracted directory
    try:
        shutil.rmtree(account_dir)
        print(f"🧹 Cleaned up temporary directory: {account_dir}")
    except Exception as e:
        print(f"⚠️  Could not clean up {account_dir}: {e}")
    
    # Show results
    print(f"\n🎉 Import completed!")
    print("=" * 40)
    print(f"📁 Files imported: {len(files_imported)}")
    for file in files_imported:
        print(f"  ✅ {file}")
    
    print(f"\n🚀 Ready to use your account!")
    print("Next steps:")
    print(f"1. Start your node: ./pouw-cli start --node-id {node_id}")
    print(f"2. Check your balance: ./pouw-cli balance --node-id {node_id}")
    print(f"3. Check mining status: ./pouw-cli status --node-id {node_id}")
    
    return True

if __name__ == "__main__":
    import sys
    import datetime
    
    # Get archive path from command line if provided
    archive_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        success = import_account(archive_path)
        if success:
            print("\n✅ Account import successful!")
        else:
            print("\n❌ Account import failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1) 