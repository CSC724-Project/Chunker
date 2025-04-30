#!/bin/bash
# Build.sh
# This script runs the commands into an automated setup for BeeGFS.
# It is assumed that the script is run in a non-interactive environment.

# The script is intended to be run as a root user or with sudo privileges.

# The script performs the following tasks:
# 1. Updates the package repository information.
# 2. Installs required packages and dependencies.
# 3. Clones the BeeGFS repository from GitHub.
# 4. Builds the BeeGFS Debian packages.
# 5. Installs the generated packages.
# 6. Sets up the BeeGFS management daemon and other services.
# 7. Configures the metadata and storage services.
# 8. Starts the services and checks their status.
# 9. Runs diagnostic commands to check the status of the BeeGFS setup.
# 10. Cleans up temporary files and directories.

# Note: The script assumes that the user has sudo privileges and can run commands as root.
# The script is intended for use on a Debian-based system (e.g., Ubuntu).


# The following interactive command was used to open a root shell;
# it is commented out since it waits for interactive input.
# sudo -i


# Launching VS Code (interactive; uncomment if required)
# code .

# Update package repository information
sudo apt update

# Clone the repository
git clone https://github.com/ThinkParQ/beegfs.git
cd beegfs/

# Install build essentials and kernel headers
sudo apt install build-essential linux-headers-$(uname -r)

# Install additional required packages
sudo apt install --no-install-recommends devscripts equivs

# Get the Ubuntu version release (e.g., "24.04", "22.04")
ubuntu_version=$(lsb_release -rs)
# Extract the major version number (e.g., "24" from "24.04")
major_version=$(echo "$ubuntu_version" | cut -d. -f1)

echo "Detected Ubuntu major version: $major_version"

if [ "$major_version" = "24" ]; then
    echo "Installing packages for Ubuntu 24..."
    sudo apt install build-essential autoconf automake pkg-config devscripts debhelper libtool libattr1-dev xfslibs-dev lsb-release kmod \
                     librdmacm-dev libibverbs-dev default-jdk zlib1g-dev libssl-dev libcurl4-openssl-dev libblkid-dev uuid-dev \
                     libnl-3-200 libnl-3-dev libnl-genl-3-200 libnl-route-3-200 libnl-route-3-dev dh-dkms
elif [ "$major_version" = "22" ]; then
    echo "Installing packages for Ubuntu 22..."
    sudo apt install build-essential autoconf automake pkg-config devscripts debhelper libtool libattr1-dev xfslibs-dev lsb-release kmod \
                     librdmacm-dev libibverbs-dev default-jdk zlib1g-dev libssl-dev libcurl4-openssl-dev libblkid-dev uuid-dev \
                     libnl-3-200 libnl-3-dev libnl-genl-3-200 libnl-route-3-200 libnl-route-3-dev dkms
else
    echo "Unsupported Ubuntu version: $ubuntu_version"
fi

# Build the Debian packages
make package-deb PACKAGE_DIR=packages DEBUILD_OPTS="-j2" #-j2 is for parallel build change it to -j4 or -j8 for more parallelism as per your CPU cores

# The following command is used to build the packages.
sudo dpkg -i packages/beegfs-common_*.deb
sudo dpkg -i packages/beegfs-utils_*.deb
sudo dpkg -i packages/beegfs-mgmtd_*.deb
sudo dpkg -i packages/beegfs-meta_*.deb
sudo dpkg -i packages/beegfs-storage_*.deb
sudo dpkg -i packages/beegfs-helperd_*.deb
sudo dpkg -i packages/beegfs-client_*.deb


#####
###
# Everything after this point goes in the README file.
###
#####

# Editing configuration files (requires interactive text editors).
# Uncomment these lines if you wish to manually adjust configuration files.
# sudo nano beegfs-client-autobuild.conf
# sudo nano beegfs-client.conf
# sudo nano beegfs-helperd.conf
# sudo nano beegfs-meta.conf
# sudo nano beegfs-mgmtd.conf
# sudo nano beegfs-storage.conf


hostname -i # This command will give you the IP address of the host.

sudo /opt/beegfs/sbin/beegfs-setup-mgmtd -p /data/beegfs/beegfs_mgmtd

# Remove any existing management directory if necessary
rm -rf /data/beegfs/beegfs_mgmtd
sudo rm -rf /data/beegfs/beegfs_mgmtd

# Run the management daemon directly
/data/beegfs/beegfs_mgmtd
ls /data/beegfs/

# Re-run the management setup command
sudo /opt/beegfs/sbin/beegfs-setup-mgmtd -p /data/beegfs/beegfs_mgmtd

# Setup metadata service with given parameters (IP may need to be adjusted)
sudo /opt/beegfs/sbin/beegfs-setup-meta -p /data/beegfs/beegfs_meta -s 2 -m <YOUR_IP_ADDRESS>

# Setup storage service (adjust parameters and paths as needed)
sudo /opt/beegfs/sbin/beegfs-setup-storage -p /mnt/myraid1/beegfs_storage -s 3 -i 301 -m <YOUR_IP_ADDRESS>

# Setup client service
sudo /opt/beegfs/sbin/beegfs-setup-client -m <YOUR_IP_ADDRESS>

# Start the services
sudo systemctl start beegfs-mgmtd
sudo systemctl status beegfs-mgmtd
sudo systemctl status beegfs-meta
sudo systemctl start beegfs-meta
sudo systemctl start beegfs-storage
sudo systemctl start beegfs-helperd
sudo systemctl start beegfs-client
sudo systemctl status beegfs-client

# Run some diagnostic commands
beegfs-ctl --listnodes --nodetype=meta --nicdetails
beegfs-ctl --listnodes --nodetype=storage --nicdetails
beegfs-ctl --listnodes --nodetype=mgmtd --nicdetails
beegfs-ctl --listnodes --nodetype=client --nicdetails

beegfs-net
beegfs-check-servers
beegfs-df