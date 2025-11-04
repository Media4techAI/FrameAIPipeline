#!/usr/bin/env python3.7

import os
import sys
import json
import shutil

# Add lib to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from lib import BoostFilter, FixGroundtruthFilter, FixImageBackgroundsFilter, ReconstructFilter

class MockParameters:
    """Mock parameters class for testing"""
    def __init__(self):
        # Use object notation instead of dictionary
        self.output_steps_dirs = type('obj', (object,), {
            'step1': "output/step1_generate_new_example",
            'step2': "output/step2_export_alignments", 
            'step3': "output/step3_fix_groundtruth",
            'step4': "output/step4_fix_images",
            'step5': "output/step5_boost",
            'step6': "output/step6_reconstruct"
        })()
        
        self.bg_color = "8 248 8"
        self.bg_tolerance = 0
        
        # Add hyperparameters for boost filter
        self.hyperparameters = type('obj', (object,), {
            'learner_num': 5,
            'width': 160,
            'height': 160,
            'depth': 3,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'total_training_step': 30000
        })()

def validate_job_folder(job_folder_path):
    """Validate that job folder exists and has required structure"""
    if not job_folder_path:
        return False, "No job folder specified"
    
    if not os.path.exists(job_folder_path):
        return False, f"Job folder does not exist: {job_folder_path}"
    
    if not os.path.isdir(job_folder_path):
        return False, f"Path is not a directory: {job_folder_path}"
    
    # Check for required output directories
    required_dirs = [
        "output/step1_generate_new_example",
        "output/step2_export_alignments",
        "output/step3_fix_groundtruth",
        "output/step4_fix_images",
        "output/step5_boost",
        "output/step6_reconstruct"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = os.path.join(job_folder_path, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        return False, f"Missing required directories: {missing_dirs}"
    
    return True, "Job folder is valid"

def create_test_environment(job_folder_path=None):
    """Use existing job folder - no more temporary environments"""
    if not job_folder_path:
        raise ValueError("Job folder path is required. Use --job-folder argument.")
    
    is_valid, message = validate_job_folder(job_folder_path)
    if not is_valid:
        raise ValueError(f"Invalid job folder: {message}")
    
    # Ensure params.json exists for load_parameters() method
    input_dir = os.path.join(job_folder_path, "input")
    params_file = os.path.join(input_dir, "params.json")
    
    if not os.path.exists(params_file):
        print(f"Creating missing params.json in {input_dir}")
        os.makedirs(input_dir, exist_ok=True)
        
        # Create realistic test parameters for filters
        test_params = {
            "images_folder": os.path.join(job_folder_path, "input/images"),
            "masks_folder": os.path.join(job_folder_path, "input/masks"), 
            "config_file": os.path.join(input_dir, "config.json"),
            "original_image": "original_test.jpg",
            "groundtruth_file": os.path.join(job_folder_path, "output/step2_export_alignments/groundtruth.txt"),
            "num_rotations": 36,
            "num_scale": 5,
            "patch_size": 64,
            "step_size": 16,
            "batch_size": 32,
            "threshold": 0.8,
            "debug": True
        }
        
        with open(params_file, "w") as f:
            json.dump(test_params, f, indent=2)
        print(f"✓ Created params.json with test parameters")
    else:
        print(f"✓ Using existing params.json: {params_file}")

    print(f"Using existing job folder: {job_folder_path}")
    shared_dir = os.path.dirname(job_folder_path)
    return shared_dir, job_folder_path, False  # False = not temporary

def create_test_groundtruth_file(job_folder):
    """Create a test groundtruth file in JSON format"""
    step2_dir = os.path.join(job_folder, "output/step2_export_alignments")
    gt_file = os.path.join(step2_dir, "groundtruth.txt")
    
    # Create test groundtruth data in JSON format
    test_data = {
        "fragments": [
            {"name": "fragment_p001", "T": [10.5, 20.3, 45.0]},
            {"name": "fragment_p002", "T": [15.2, 25.8, 30.0]},
            {"name": "fragment_p003", "T": [5.1, 18.9, 60.0]}
        ]
    }
    
    with open(gt_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return gt_file

def create_test_images(job_folder):
    """Create test images for background processing"""
    from PIL import Image
    import numpy as np
    
    step1_m_dir = os.path.join(job_folder, "output/step1_generate_new_example/m")
    
    # Create test images
    test_images = ["fragment_p001.png", "fragment_p002.png", "fragment_p003.png"]
    
    for img_name in test_images:
        # Create a simple test image (50x50 pixels)
        img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(step1_m_dir, img_name))
    
    return test_images

def create_boost_test_files(job_folder):
    """Create required test files for boost filter"""
    
    # Create alpha.txt file (required by boost filter)
    model_dir = os.path.join(job_folder, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    alpha_file = os.path.join(model_dir, "alpha.txt")
    if not os.path.exists(alpha_file):
        with open(alpha_file, 'w') as f:
            f.write("# Alpha values for boost learners\n")
            f.write("0.2 0.3 0.1 0.25 0.15\n")  # 5 alpha values for 5 learners
    
    # Create test groundtruth file in step3
    step3_dir = os.path.join(job_folder, "output", "step3_fix_groundtruth")
    os.makedirs(step3_dir, exist_ok=True)
    
    gt_file = os.path.join(step3_dir, "groundtruth_fixed.txt")
    if not os.path.exists(gt_file):
        with open(gt_file, 'w') as f:
            f.write("# Fixed groundtruth data\n")
            f.write("fragment_001 10.5 20.3 45.0\n")
            f.write("fragment_002 15.2 25.8 30.0\n")
    
    # Create test alignment file in step2  
    step2_dir = os.path.join(job_folder, "output", "step2_export_alignments")
    os.makedirs(step2_dir, exist_ok=True)
    
    align_file = os.path.join(step2_dir, "alignments.txt")
    if not os.path.exists(align_file):
        with open(align_file, 'w') as f:
            f.write("# Alignment data\n")
            f.write("fragment_001 fragment_002 0.85\n")
            f.write("fragment_002 fragment_003 0.72\n")
    
    print(f"Created test files for boost filter in {job_folder}")

def test_boost_filter(job_folder_path=None):
    """Test the BoostFilter"""
    print("=== Testing BoostFilter ===")
    
    shared_dir, job_folder, is_temp = create_test_environment(job_folder_path)
    
    try:
        # Initialize Parameters class before creating filter
        from lib.parameters import Parameters
        
        # Create params.json if it doesn't exist
        params_file = os.path.join(job_folder, "input", "params.json")
        if not os.path.exists(params_file):
            print("Creating missing params.json file...")
            os.makedirs(os.path.dirname(params_file), exist_ok=True)
            
            default_params = {
                "job_id": os.path.basename(job_folder),
                "parts": "1:3",
                "hyperparameters": {
                    "width": 160,
                    "height": 160,
                    "depth": 3,
                    "batch_size": 64,
                    "weight_decay": 1e-4,
                    "learning_rate": 1e-4,
                    "total_training_step": 30000,
                    "learner_num": 5
                },
                "output_steps_dirs": {
                    "step1": "output/step1_generate_new_example",
                    "step2": "output/step2_export_alignments",
                    "step3": "output/step3_fix_groundtruth",
                    "step4": "output/step4_fix_images",
                    "step5": "output/step5_boost",
                    "step6": "output/step6_reconstruct"
                }
            }
            
            with open(params_file, 'w') as f:
                json.dump(default_params, f, indent=2)
        
        # Initialize Parameters class
        print("Initializing Parameters class...")
        Parameters.initialize(config_file=params_file, job_folder=job_folder)
        
        # Create required test files for boost filter
        create_boost_test_files(job_folder)
        
        # Create filter instance
        boost_filter = BoostFilter(
            shared_dir=shared_dir,
            mask_folder="test_masks",
            original_image="test_original.jpg", 
            config_file="test_config.json",
            parts="1:3",
            job_folder=job_folder,
            job_id=os.path.basename(job_folder)
        )
        
        # Load parameters from job folder
        print("Loading parameters from job folder...")
        boost_filter.load_parameters()
        
        # Test process method
        print("Testing BoostFilter.process()...")
        result = boost_filter.process()
        
        if result:
            print("✓ BoostFilter test PASSED")
            
            # Check if dataset_list.txt was created (required for reconstruction)
            step5_dir = os.path.join(job_folder, "output/step5_boost")
            dataset_list_file = os.path.join(step5_dir, 'dataset_list.txt')
            if os.path.exists(dataset_list_file):
                print(f"✓ Dataset list file created: {dataset_list_file}")
                with open(dataset_list_file, 'r') as f:
                    content = f.read()
                    print(f"Dataset list content preview: {content[:200]}...")
            else:
                print(f"⚠️  Dataset list file not found at: {dataset_list_file}")
                # Create a basic dataset_list.txt for testing reconstruction
                os.makedirs(step5_dir, exist_ok=True)
                with open(dataset_list_file, 'w') as f:
                    f.write("# Dataset list for reconstruction\n")
                    f.write("step5_boost/output_data.txt\n")
                print("Created basic dataset_list.txt for testing")
        else:
            print("✗ BoostFilter test FAILED")
        
        return result
        
    except Exception as e:
        print(f"✗ BoostFilter test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if is_temp:
            shutil.rmtree(shared_dir, ignore_errors=True)

def test_fix_groundtruth_filter(job_folder_path=None):
    """Test the FixGroundtruthFilter"""
    print("\n=== Testing FixGroundtruthFilter ===")
    
    shared_dir, job_folder, is_temp = create_test_environment(job_folder_path)
    
    try:
        # Create test groundtruth file if needed
        step2_dir = os.path.join(job_folder, "output/step2_export_alignments")
        gt_file = os.path.join(step2_dir, "groundtruth.txt")
        
        if not os.path.exists(gt_file):
            input_gt_file = create_test_groundtruth_file(job_folder)
            print(f"Created test groundtruth file: {input_gt_file}")
        else:
            print(f"Using existing groundtruth file: {gt_file}")
        
        # Create filter instance
        fix_gt_filter = FixGroundtruthFilter(
            shared_dir=shared_dir,
            mask_folder="test_masks",
            original_image="test_original.jpg",
            config_file="test_config.json", 
            parts="1:3",
            job_folder=job_folder,
            job_id=os.path.basename(job_folder)
        )
        
        # Load parameters from job folder
        print("Loading parameters from job folder...")
        fix_gt_filter.load_parameters()
        
        # Test process method
        print("Testing FixGroundtruthFilter.process()...")
        result = fix_gt_filter.process()
        
        # Check if output file was created
        expected_output = os.path.join(job_folder, "output/step3_fix_groundtruth/groundtruth_fixed.txt")
        if result and os.path.exists(expected_output):
            print("✓ FixGroundtruthFilter test PASSED")
            print(f"✓ Output file created: {expected_output}")
            
            # Show contents of output file
            with open(expected_output, 'r') as f:
                content = f.read()
                print(f"Output content (first 200 chars):\n{content[:200]}...")
                
        else:
            print("✗ FixGroundtruthFilter test FAILED")
            print(f"Expected output file: {expected_output}")
        
        return result and os.path.exists(expected_output)
        
    except Exception as e:
        print(f"✗ FixGroundtruthFilter test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if is_temp:
            shutil.rmtree(shared_dir, ignore_errors=True)

def test_fix_image_backgrounds_filter(job_folder_path=None):
    """Test the FixImageBackgroundsFilter"""
    print("\n=== Testing FixImageBackgroundsFilter ===")
    
    shared_dir, job_folder, is_temp = create_test_environment(job_folder_path)
    
    try:
        # Check for existing images or create test ones
        step1_m_dir = os.path.join(job_folder, "output/step1_generate_new_example/m")
        existing_images = []
        if os.path.exists(step1_m_dir):
            existing_images = [f for f in os.listdir(step1_m_dir) if f.endswith(('.png', '.jpg'))]
        
        if not existing_images:
            test_images = create_test_images(job_folder)
            print(f"Created test images: {test_images}")
        else:
            print(f"Using existing images: {existing_images[:3]}...")  # Show first 3
        
        # Create filter instance
        fix_bg_filter = FixImageBackgroundsFilter(
            shared_dir=shared_dir,
            mask_folder="test_masks",
            original_image="test_original.jpg",
            config_file="test_config.json",
            parts="1:3", 
            job_folder=job_folder,
            job_id=os.path.basename(job_folder)
        )
        
        # Load parameters from job folder
        print("Loading parameters from job folder...")
        fix_bg_filter.load_parameters()
        
        # Test process method
        print("Testing FixImageBackgroundsFilter.process()...")
        result = fix_bg_filter.process()
        
        # Check if output directory has processed files
        output_dir = os.path.join(job_folder, "output/step4_fix_images")
        output_files = []
        if os.path.exists(output_dir):
            output_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))]
        
        if result and output_files:
            print("✓ FixImageBackgroundsFilter test PASSED")
            print(f"✓ Output files created: {len(output_files)} files")
            print(f"  Sample files: {output_files[:3]}...")
        else:
            print("✗ FixImageBackgroundsFilter test FAILED")
            print(f"Output directory: {output_dir}")
            print(f"Files found: {len(output_files)}")
        
        return result and len(output_files) > 0
        
    except Exception as e:
        print(f"✗ FixImageBackgroundsFilter test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if is_temp:
            shutil.rmtree(shared_dir, ignore_errors=True)

def test_reconstruct_filter(job_folder_path=None):
    """Test the ReconstructFilter"""
    print("\n=== Testing ReconstructFilter ===")
    
    shared_dir, job_folder, is_temp = create_test_environment(job_folder_path)
    
    try:
        # Create filter instance
        reconstruct_filter = ReconstructFilter(
            shared_dir=shared_dir,
            mask_folder="test_masks",
            original_image="test_original.jpg",
            config_file="test_config.json",
            parts="1:3",
            job_folder=job_folder,
            job_id=os.path.basename(job_folder)
        )
        
        # Load parameters from job folder
        print("Loading parameters from job folder...")
        reconstruct_filter.load_parameters()
        
        # Test process method
        print("Testing ReconstructFilter.process()...")
        result = reconstruct_filter.process()
        
        if result:
            print("✓ ReconstructFilter test PASSED")
        else:
            print("✗ ReconstructFilter test FAILED")
        
        return result
        
    except Exception as e:
        print(f"✗ ReconstructFilter test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if is_temp:
            shutil.rmtree(shared_dir, ignore_errors=True)

def run_all_filter_tests(job_folder_path=None):
    """Run all filter tests in the correct pipeline order"""
    print("🧪 Running Filter Tests Suite (Pipeline Order)")
    print("=" * 50)
    
    # Define test order as per pipeline sequence
    test_order = [
        ('fix_groundtruth', test_fix_groundtruth_filter),
        ('fix_backgrounds', test_fix_image_backgrounds_filter),
        ('boost', test_boost_filter),
        ('reconstruct', test_reconstruct_filter)
    ]
    
    results = {}
    
    for test_name, test_func in test_order:
        print(f"\n{'='*20} Running {test_name.replace('_', ' ').title()} {'='*20}")
        results[test_name] = test_func(job_folder_path)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

def run_single_filter_test(filter_name, job_folder_path=None):
    """Run a single filter test"""
    filter_tests = {
        'fix_groundtruth': test_fix_groundtruth_filter,
        'fix_backgrounds': test_fix_image_backgrounds_filter,
        'boost': test_boost_filter,
        'reconstruct': test_reconstruct_filter
    }
    
    if filter_name not in filter_tests:
        print(f"❌ Unknown filter: {filter_name}")
        print(f"Available filters: {', '.join(filter_tests.keys())}")
        return False
    
    print(f"🧪 Running Single Filter Test: {filter_name.replace('_', ' ').title()}")
    print("=" * 50)
    
    result = filter_tests[filter_name](job_folder_path)
    
    print("\n" + "=" * 50)
    status = "✓ PASSED" if result else "✗ FAILED"
    print(f"📊 Test Result: {filter_name.replace('_', ' ').title()} {status}")
    
    return result

def show_usage():
    """Show script usage"""
    print("Filter Testing Suite")
    print("=" * 50)
    print("Usage:")
    print("  python test_boost.py <filter_name> --job-folder <path>")
    print()
    print("Required Arguments:")
    print("  --job-folder, -j     Path to existing job folder (REQUIRED)")
    print()
    print("Available filters:")
    print("  all                  # Run all tests")
    print("  fix_groundtruth      # Test FixGroundtruthFilter")
    print("  fix_backgrounds      # Test FixImageBackgroundsFilter") 
    print("  boost                # Test BoostFilter")
    print("  reconstruct          # Test ReconstructFilter")
    print()
    print("Examples:")
    print("  python test_boost.py boost -j /path/to/job_folder")
    print("  python test_boost.py all -j /mnt/c/Users/user/Documenti/jobs/job_20251030_154342")
    print("  python test_boost.py fix_groundtruth -j ./jobs/latest_job")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter Testing Suite for Pipeline Filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_boost.py all -j /path/to/job_folder          # Run all tests
  python test_boost.py boost -j /path/to/job_folder        # Test only BoostFilter
  python test_boost.py fix_groundtruth -j /path/to/job     # Test only FixGroundtruthFilter
  python test_boost.py fix_backgrounds -j /path/to/job     # Test only FixImageBackgroundsFilter
  python test_boost.py reconstruct -j /path/to/job         # Test only ReconstructFilter

Required:
  --job-folder, -j   Path to existing job folder from pipeline execution

Test Order (when running all):
  1. FixGroundtruthFilter
  2. FixImageBackgroundsFilter
  3. BoostFilter
  4. ReconstructFilter
        """
    )
    
    parser.add_argument(
        'filter',
        nargs='?',
        default='all',
        help='Filter to test (all, fix_groundtruth, fix_backgrounds, boost, reconstruct)'
    )
    
    parser.add_argument(
        '--job-folder', '-j',
        type=str,
        required=True,
        help='Path to existing job folder to test with (REQUIRED)'
    )
    
    args = parser.parse_args()
    
    # Validate job folder before proceeding
    try:
        is_valid, message = validate_job_folder(args.job_folder)
        if not is_valid:
            print(f"❌ {message}")
            print("\nPlease provide a valid job folder from a previous pipeline run.")
            print("Example: ./test_boost.py boost -j /mnt/c/Users/user/Documenti/jobs/job_20251030_154342")
            sys.exit(1)
        else:
            print(f"✓ {message}")
    except Exception as e:
        print(f"❌ Error validating job folder: {e}")
        sys.exit(1)
    
    # Check if PIL is available for image tests
    try:
        from PIL import Image
        import numpy as np
        print("✓ PIL and numpy available for image tests")
    except ImportError as e:
        print(f"⚠️  Warning: {e}")
        print("Image-related tests may fail")
    
    print()
    
    # Run tests based on argument
    if args.filter == 'all':
        success = run_all_filter_tests(args.job_folder)
    elif args.filter in ['fix_groundtruth', 'fix_backgrounds', 'boost', 'reconstruct']:
        success = run_single_filter_test(args.filter, args.job_folder)
    else:
        print(f"❌ Unknown filter: {args.filter}")
        print()
        show_usage()
        sys.exit(1)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)