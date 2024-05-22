pulseaudio --start

INSTALLDIR="/opt/intel/openvino_2022.3.1"
export INTEL_OPENVINO_DIR="$INSTALLDIR"
export InferenceEngine_DIR=$INSTALLDIR/runtime/cmake
export ngraph_DIR=$INSTALLDIR/runtime/cmake
export OpenVINO_DIR=$INSTALLDIR/runtime/cmake
export HDDL_INSTALL_DIR="/opt/intel/openvino_2022.3.1/runtime/3rdparty/hddl"
export PKG_CONFIG_PATH="/opt/intel/openvino_2022.3.1/runtime/lib/aarch64/pkgconfig"
export LD_LIBRARY_PATH="/opt/intel/openvino_2022.3.1/tools/compile_tool:/opt/intel/openvino_2022.3.1/runtime/3rdparty/hddl/lib:/opt/intel/openvino_2022.3.1/runtime/lib/aarch64"
export PYTHONPATH="/opt/intel/openvino_2022.3.1/python/python3.9:"

cd /root/server/
/root/miniconda3/bin/python /root/server/server.py 
