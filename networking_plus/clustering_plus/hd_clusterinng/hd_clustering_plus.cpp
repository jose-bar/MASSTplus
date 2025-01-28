#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include "hd_tools.cpp"
//#include "bits/stdc++.h"
using namespace std;

#define FOR(i, hi) for (int i = 0; i < hi; ++i)
#define FORR(i, hi) for (int i = hi - 1; i >= 0; --i)
#define For(i, lo, hi) for (int i = lo; i < hi; ++i)
#define Forr(i, hi, lo) for (int i = hi - 1; i >= lo; --i)

typedef struct {
    int peak_idx;
    int scan;
    int local_cluster_idx;
    double pepmass;
    double true_mz;
    double mass;
    HDVector hd_vec;  // Add HD vector representation
} SpectrumInfo;

struct PeakInfo {
    int mz_bin;
    int mass_bin;
    int scan;
    double true_mz;
    double mass;
    HDVector hd_vec;
    
    PeakInfo() = default;
    PeakInfo(const PeakInfo&) = default;
    PeakInfo& operator=(const PeakInfo&) = default;
    ~PeakInfo() = default;
};

typedef struct {
    double pepmass;
    double RT;
    long double magnitude;
} Spec_info ;

typedef struct {
    int a, b;
    double massprod;
} Match;

typedef struct {
    int scan;
    double pepmass;
    double mz;
} top_four_Info;

typedef struct {
    int ind;
    double mass;
    double mz;
} Top_peak_t;

/*peakstorage datastructure**************************************************************/

//typedef vector<vector<SpectrumInfo>> mztospectra_t;
/*index: the mz value bin index; value: the peaks with a mz value within the threshold*/

typedef vector<vector<PeakInfo>> mztospectra_t;

typedef unordered_map<int, vector<PeakInfo>> spectratomz_t;
/*index: the scan-number of each spectra; value: the peaks of that scan in the bin*/

typedef unordered_map<int, Spec_info> spectra_general_info_t;
/*index: the scan-number of each spectra; value: the general peak information (pepmass, RT) of that scan in the bin*/

typedef pair<spectratomz_t, spectra_general_info_t> spectratoinfo_t; 
/*the combination of peak informatin and general information for each scan*/


/*topfour datastructure ******************************************************************/

typedef unordered_map<int, vector<Top_peak_t>> scan_to_top_peaks_t;
/*index int: the scan-number of each spectra; values: a vector containing the top-four peaks of the scan*/

//typedef vector<top_four_Info> mz_to_topfour_t;
typedef vector<vector<int>> mz_to_topfour_t;
/*index int: the mz value of the topfour peaks; values: a vector containing the scans with at least one top-four peak in that bin*/

typedef vector<mz_to_topfour_t> pepmass_spectra_t;
/*index int: pepmass bins; values: the mz_to_topfour peaks vector with index as the mz bin value and the information stored as top-four peaks*/

typedef pair<pepmass_spectra_t, scan_to_top_peaks_t> top_four_combined_t;


/*cluster storage results*****************************************************************/
typedef unordered_map<int, int> scan_to_clustercenter_t;
/*index int: the cluster-center int; value: a vector containing the scans in that cluster; the first scan number refers to the cluster-center scan*/

typedef vector<vector<int>> clustercenter_to_scans_t;
/*index int: the scan number of spectras; value: the cluster-center of the scan*/

typedef pair<scan_to_clustercenter_t, clustercenter_to_scans_t> cluster_info_t;
/*the combination of the above datastructures*/

typedef pair<spectratoinfo_t, top_four_combined_t> topfour_pepmass_raw_t;

typedef vector<vector<int>> pepmass_distribution_t;

typedef pair<unordered_map<string, int>, unordered_map<int, string>> file_info_t;

static double TOLERANCE = 0.01;
static double THRESHOLD = 0.7;
static double MASSTOLERANCE = 1.0;
static double TOPFOURTOLERANCE = 0.1;
static int THRESHOLD_TOPFOUR = 5000;
static int THRESHOLD_INDEXING = 10;
static bool INDEXING_CHOICE = true;
static bool TOPFOUR_CHOICE = true;
static double SELECTIONRANGE = 50.0;
static int Output_cluster_minsize = 2;
static int FILTERK = 5;

vector<string> get_files(const string& content_files) {
    vector<string> filenames;
    ifstream f(content_files);
    
    if (!f.is_open()) {
        cerr << "ERROR: Could not open file: " << content_files << endl;
        return filenames;
    }
    
    string path_string;
    filenames.reserve(100); // Reserve space to avoid reallocations
    
    while (getline(f, path_string)) {
        size_t ext_begin = path_string.find_last_of(".");
        if (ext_begin == string::npos) continue;
        
        string_view file_ext(path_string.data() + ext_begin + 1);
        if (file_ext == "mgf" || file_ext == "mzML") {
            filenames.push_back(std::move(path_string));
        }
    }
    return filenames;
}


bool sort_tuple(const tuple<int, double, double>& a, const tuple<int, double, double>& b) {
    return (get<2>(a) > get<2>(b));
}

void cleanup_spectratomz(spectratomz_t& spectratomz) {
    spectratomz.clear();
}


pair<file_info_t, pair<pepmass_distribution_t, topfour_pepmass_raw_t>> parse_inputs(vector<string>& filenames) {
    int bin_len = get_dim();
    float bin_size = TOLERANCE;

    // Initialize return structures
    pair<file_info_t, pair<pepmass_distribution_t, topfour_pepmass_raw_t>> res;
    file_info_t& file_info = res.first;
    pepmass_distribution_t& pepmass_distribution = res.second.first;
    topfour_pepmass_raw_t& topfour_pepmass_raw = res.second.second;
    
    unordered_map<string, int>& file_start_scan = file_info.first;
    unordered_map<int, string>& scan_file_src = file_info.second;
    
    spectratoinfo_t& spectratoinfo_shared = topfour_pepmass_raw.first;
    spectratomz_t& spectratomz_shared = spectratoinfo_shared.first;
    spectra_general_info_t& spectra_general = spectratoinfo_shared.second;
    
    top_four_combined_t& top_four_combined = topfour_pepmass_raw.second;
    pepmass_spectra_t& pepmass_spectra = top_four_combined.first;
    scan_to_top_peaks_t& scan_to_top_peaks = top_four_combined.second;

    try {
        // Generate and pack HD vectors
        unique_ptr<float[]> lv_hvs(generate_level_vectors(HD_DIM, HD_Q));
        unique_ptr<float[]> id_hvs(generate_id_vectors(HD_DIM, bin_len, HD_ID_FLIP));
        unique_ptr<uint32_t[]> packed_lv(new uint32_t[(HD_Q+1) * pack_len]);
        unique_ptr<uint32_t[]> packed_id(new uint32_t[bin_len * pack_len]);

        bit_packing(packed_lv.get(), lv_hvs.get(), HD_DIM, pack_len, HD_Q+1);
        bit_packing(packed_id.get(), id_hvs.get(), HD_DIM, pack_len, bin_len);

        pepmass_distribution.resize(max(500, bin_len));
        pepmass_spectra.resize(max(500, bin_len));

        int scan_tracer = 0;

        // Process each input file
        for (const string& filename : filenames) {
            file_start_scan[filename] = scan_tracer;
            string file_ext = filename.substr(filename.find_last_of(".") + 1);
            
            if (file_ext == "mgf") {
                ifstream f(filename);
                if (!f.is_open()) {
                    throw runtime_error("Could not open MGF file: " + filename);
                }

                string line;
                while (getline(f, line)) {
                    if (line == "BEGIN IONS") {
                        double pepmass = -1;
                        double rtinsec = 0.0;
                        vector<double> mz_values, intensities;
                        
                        while (getline(f, line)) {
                            if (line.empty()) continue;
                            
                            // Improved PEPMASS parsing
                            if (line.rfind("PEPMASS=", 0) == 0) {
                                try {
                                    string pepmass_str = line.substr(8);
                                    // Remove any trailing whitespace
                                    pepmass_str.erase(pepmass_str.find_last_not_of(" \n\r\t") + 1);
                                    pepmass = stod(pepmass_str);
                                } catch (const std::exception& e) {
                                    cerr << "Error parsing PEPMASS: " << line << endl;
                                    continue;
                                }
                            }
                            else if (line.rfind("RTINSECONDS=", 0) == 0) {
                                try {
                                    string rt_str = line.substr(11);
                                    // Remove any whitespace and handle scientific notation
                                    rt_str.erase(remove_if(rt_str.begin(), rt_str.end(), ::isspace), rt_str.end());
                                    size_t lastPos;
                                    rtinsec = stod(rt_str, &lastPos);
                                    if (lastPos != rt_str.length()) {
                                        cerr << "Warning: Extra characters in RTINSECONDS value: " << rt_str << endl;
                                    }
                                } catch (const std::exception& e) {
                                    // cerr << "Error parsing RTINSECONDS: " << line << endl;
                                    // cerr << "Detailed error: " << e.what() << endl;
                                    // Use 0.0 as fallback value instead of skipping
                                    rtinsec = 0.0;
                                }
                            }
                            else if (line == "END IONS") break;
                            else {
                                // Parse mz and intensity values
                                try {
                                    stringstream ss(line);
                                    double mz, intensity;
                                    if (ss >> mz >> intensity) {
                                        mz_values.push_back(mz);
                                        intensities.push_back(intensity);
                                    }
                                } catch (const std::exception& e) {
                                    cerr << "Error parsing mz/intensity line: " << line << endl;
                                    continue;
                                }
                            }
                        }

                        // Process spectrum if valid
                        if (pepmass != -1 && !mz_values.empty()) {
                            unique_ptr<uint32_t[]> hd_packed(new uint32_t[pack_len]);
                            encode_spectrum_hd(hd_packed.get(), mz_values, intensities,
                                            packed_lv.get(), packed_id.get(),
                                            HD_DIM, HD_Q, pack_len, bin_size);

                            vector<PeakInfo> peaks;
                            for (size_t i = 0; i < mz_values.size(); i++) {
                                PeakInfo peak;
                                peak.mz_bin = int(mz_values[i] / bin_size);
                                peak.mass_bin = int(intensities[i] * HD_Q);
                                peak.true_mz = mz_values[i];
                                peak.mass = intensities[i];
                                peak.scan = scan_tracer;
                                peak.hd_vec = HDVector(pack_len);  // Create HDVector with proper size
                                memcpy(peak.hd_vec.data(), hd_packed.get(), pack_len * sizeof(uint32_t));
                                peaks.push_back(peak);
                            }
                                                        
                            spectratomz_shared[scan_tracer] = peaks;
                            
                            int pepmass_idx = round(pepmass / MASSTOLERANCE);
                            if (pepmass_idx >= pepmass_distribution.size()) {
                                pepmass_distribution.resize(pepmass_idx * 2);
                                pepmass_spectra.resize(pepmass_idx * 2);
                            }
                            
                            pepmass_distribution[pepmass_idx].push_back(scan_tracer);
                            spectra_general[scan_tracer] = {pepmass, rtinsec, 0.0};
                            scan_file_src[scan_tracer] = filename;

                            scan_tracer++;
                        }
                    }
                }
            }
        }
        return res;
    }
    catch (...) {
        cleanup_spectratomz(spectratomz_shared);
        throw;
    }
}


struct scan_comp {
    inline bool operator() (const SpectrumInfo& s1, const SpectrumInfo& s2) {
        return (s1.scan < s2.scan);
    }
    inline bool operator() (const SpectrumInfo& s1, double scan) {
        return (s1.scan < scan);
    }
    inline bool operator() (double scan, const SpectrumInfo& s2) {
        return (scan < s2.scan);
    }
};

bool cmp(Match x, Match y) {
    return (x.massprod > y.massprod);
}

int brute_force_clustering(spectratoinfo_t& spectrainfo_all, vector<int>& scan_numbers_in_bin, cluster_info_t& cluster_info) {
    if (scan_numbers_in_bin.size() == 0) {
        return 0;
    }
    int total_pairwise_counter = 0;
    spectratomz_t &spectratomz_shared = spectrainfo_all.first;
    spectra_general_info_t &spectra_general = spectrainfo_all.second;

    scan_to_clustercenter_t &spectra_cluster = cluster_info.first;
    clustercenter_to_scans_t &cluster_content = cluster_info.second;

    unordered_map<int, bool> cluster_marker;
    spectratomz_t local_spectratomz_center;
    for (int scan_idx = 0; scan_idx < scan_numbers_in_bin.size(); scan_idx ++) {
        int fixed_scan = scan_numbers_in_bin[scan_idx];
        vector<PeakInfo>& peaks_share = spectratomz_shared[fixed_scan];
        if (spectra_cluster[fixed_scan] >= 0) {
            int cluster_idx_global = spectra_cluster[fixed_scan];
            int cluster_center_scan = cluster_content[cluster_idx_global][0];
            if (!cluster_marker[cluster_idx_global]) {
                local_spectratomz_center[cluster_idx_global] = spectratomz_shared[cluster_center_scan];
                cluster_marker[cluster_idx_global] = true;
            }
        } else {
            int cluster_belonging = -1;
            double max_prod = 0.0;
            for (auto iter = local_spectratomz_center.begin(); iter != local_spectratomz_center.end(); iter ++) {
                vector<Match> matches;
                int cluster_idx_global = iter -> first;
                vector<PeakInfo>& cluster_peaks = iter -> second;
                int cluster_center_scan = cluster_content[cluster_idx_global][0];
                if (abs(spectra_general[cluster_center_scan].pepmass - spectra_general[fixed_scan].pepmass) > MASSTOLERANCE) {
                    continue;
                }
                double product_res = 0.0;
                for (int i = 0; i < peaks_share.size(); i++) {
                    PeakInfo peak_shared = peaks_share[i];
                    for (int j = 0; j < cluster_peaks.size(); j++) {
                        PeakInfo peak_cluster = cluster_peaks[j];
                        if (abs(peak_shared.true_mz - peak_cluster.true_mz) <= TOLERANCE) {
                            product_res += peak_shared.mass * peak_shared.mass;
                        } 
                    }
                }
                if (product_res > max_prod) {
                    max_prod = product_res;
                    cluster_belonging = cluster_idx_global;
                }
            }

            if ((max_prod > 0.0) && (cluster_belonging >= 0)) {
                cluster_content[cluster_belonging].push_back(fixed_scan);
                spectra_cluster[fixed_scan] = cluster_belonging;
                cluster_marker[cluster_belonging] = true;
            } else {
                //the cluster is brand new , creating a new cluster
                int cluster_idx_global = cluster_content.size();
                //int local_cluster_idx = query_specs.size();
                //query_specs.push_back(cluster_idx_global);
                vector<int> new_cluster_sublist;
                cluster_content.push_back(new_cluster_sublist);
                cluster_content[cluster_idx_global].push_back(fixed_scan);
                spectra_cluster[fixed_scan] = cluster_idx_global;
                cluster_marker[cluster_idx_global] = true;
                local_spectratomz_center[cluster_idx_global] = peaks_share;
            }
        }
    }
    return total_pairwise_counter;
}


int cluster_local_bin(spectratoinfo_t& spectrainfo_all, vector<int>& scan_numbers_in_bin, cluster_info_t& cluster_info) {

    // cout << "Entering cluster_local_bin with " << scan_numbers_in_bin.size() << " scans" << endl;


    spectratomz_t &spectratomz_shared = spectrainfo_all.first;
    spectra_general_info_t &spectra_general = spectrainfo_all.second;
    scan_to_clustercenter_t& spectra_cluster = cluster_info.first;
    clustercenter_to_scans_t& cluster_content = cluster_info.second;
    
    unordered_map<int, bool> cluster_marker;
    const size_t INITIAL_SIZE = 200000;
    mztospectra_t local_mztospectra_center(INITIAL_SIZE);
    spectratomz_t local_spectratomz_center;
    vector<int> query_specs;
    
    cluster_marker.reserve(scan_numbers_in_bin.size());
    query_specs.reserve(scan_numbers_in_bin.size());

    for (int scan_idx : scan_numbers_in_bin) {
        int fixed_scan = scan_idx;
        //cout << "Processing scan: " << scan_idx << endl;
        vector<PeakInfo>& peaks_share = spectratomz_shared[fixed_scan];
        //cout << "Peaks size: " << peaks_share.size() << endl;
        double pepmass_scan = spectra_general[fixed_scan].pepmass;

        if (peaks_share.empty()) {
            cout << "Warning: Empty peaks for scan " << scan_idx << endl;
            continue;
        }

        if (spectra_cluster[fixed_scan] >= 0) {
           // Handle existing cluster
           int cluster_idx_global = spectra_cluster[fixed_scan];
           int cluster_center_scan = cluster_content[cluster_idx_global][0];
           
           if (!cluster_marker[cluster_idx_global]) {
               local_spectratomz_center[cluster_idx_global] = spectratomz_shared[cluster_center_scan];
               cluster_marker[cluster_idx_global] = true;
               int local_cluster_idx = query_specs.size();
               query_specs.push_back(cluster_idx_global);

               for (const PeakInfo& peak : peaks_share) {
                   int ind_peak = peak.mz_bin;
                   if (ind_peak >= local_mztospectra_center.size()) {
                       local_mztospectra_center.resize(ind_peak * 3 / 2 + 1);
                   }
                   local_mztospectra_center[ind_peak].push_back(PeakInfo{
                        peak.mz_bin,
                        peak.mass_bin,
                        fixed_scan,  // or cluster_center_scan
                        peak.true_mz,
                        peak.mass,
                        peak.hd_vec
                    });
                }
            }
        } else {
           // Find closest cluster
           int cluster_belonging = -1;
           double max_similarity = 0.0;

           for (const auto& cluster_pair : local_spectratomz_center) {
               int cluster_idx_global = cluster_pair.first;
               const vector<PeakInfo>& cluster_peaks = cluster_pair.second;
               
                // Check pepmass tolerance
                int cluster_center_scan = cluster_content[cluster_idx_global][0];
                if (abs(spectra_general[cluster_center_scan].pepmass - pepmass_scan) > MASSTOLERANCE) {
                   continue;
                }

               // Calculate HD similarity
               for (const PeakInfo& peak1 : peaks_share) {
                   for (const PeakInfo& peak2 : cluster_peaks) {
                       if (abs(peak1.true_mz - peak2.true_mz) <= TOLERANCE) {
                           double similarity = get_hd_similarity(peak1.hd_vec, peak2.hd_vec);
                           if (similarity >= THRESHOLD && similarity > max_similarity) {
                               max_similarity = similarity;
                               cluster_belonging = cluster_idx_global;
                           }
                       }
                   }
               }
           }

           if (max_similarity > 0.0 && cluster_belonging >= 0) {
               // Add to existing cluster
               cluster_content[cluster_belonging].push_back(fixed_scan);
               spectra_cluster[fixed_scan] = cluster_belonging;
               cluster_marker[cluster_belonging] = true;
           } else {
               // Create new cluster
               int cluster_idx_global = cluster_content.size();
               int local_cluster_idx = query_specs.size();
               query_specs.push_back(cluster_idx_global);
               cluster_content.emplace_back(vector<int>{fixed_scan});
               spectra_cluster[fixed_scan] = cluster_idx_global;
               cluster_marker[cluster_idx_global] = true;
               local_spectratomz_center[cluster_idx_global] = peaks_share;

               for (const PeakInfo& peak : peaks_share) {
                   int ind_peak = peak.mz_bin;
                   if (ind_peak >= local_mztospectra_center.size()) {
                       local_mztospectra_center.resize(ind_peak * 3 / 2 + 1);
                   }
                   local_mztospectra_center[ind_peak].push_back(PeakInfo{
                        peak.mz_bin,
                        peak.mass_bin,
                        fixed_scan,  // or cluster_center_scan
                        peak.true_mz,
                        peak.mass,
                        peak.hd_vec
                    });
               }
           }
       }
   }
   return 0;
}


void generate_clusters(pepmass_distribution_t &pepmass_distribution, spectratoinfo_t& spectrainfo_all, top_four_combined_t& top_four_combined, cluster_info_t& cluster_info) {
    pepmass_spectra_t &pepmass_spectra = top_four_combined.first;
    scan_to_top_peaks_t &scan_to_top_peaks = top_four_combined.second;

    scan_to_clustercenter_t& spectra_cluster = cluster_info.first;
    clustercenter_to_scans_t& cluster_content = cluster_info.second;

    spectratomz_t &spectratomz_all = spectrainfo_all.first;
    for (auto it = spectratomz_all.begin(); it != spectratomz_all.end(); ++it) {
        int query_scan = it->first;
        spectra_cluster[query_scan] = -1;
    }

    float total_time = 0.0;
    float pepmassbin_abovethresh_total = 0.0; 
    int bins_with_topfour = 0;
    //long total_pairwise_prod_topfour = 0;

    int processed_bins = 0;
    int processed_spectra = 0;
    cout << "total number of pepmass bins: " << pepmass_spectra.size() << endl;
    for (int pepmass_bin = 0; pepmass_bin < pepmass_spectra.size(); pepmass_bin ++) {
        //cout << "Processing pepmass_bin: " << pepmass_bin << endl;
        int spectra_counter = pepmass_distribution[pepmass_bin].size();
        //cout << "Spectra count in bin: " << spectra_counter << endl;
        if (spectra_counter == 0) continue;
        
        // Log before branch
        if (spectra_counter > THRESHOLD_TOPFOUR) {
            cout << "Entering topfour path" << endl;
            mz_to_topfour_t& mz_to_topfour = pepmass_spectra[pepmass_bin];
            cout << "mz_to_topfour size: " << mz_to_topfour.size() << endl;
        }
        
        processed_bins++;
        processed_spectra += spectra_counter;
        //cout << "Total processed - Bins: " << processed_bins << " Spectra: " << processed_spectra << endl;
        int spectra_counter_for_pepmass = pepmass_distribution[pepmass_bin].size(); 
        if (spectra_counter_for_pepmass > THRESHOLD_TOPFOUR) {
            //case 1: the total number of spectra in the pepmass bin is worthy to use topfour
            //int pairwise_prod_pepmass_bin = 0;
            //cout << "entering case 1 " << endl;
            auto topfour_start = chrono::steady_clock::now();
            mz_to_topfour_t& mz_to_topfour = pepmass_spectra[pepmass_bin];
            for (int top_four_bin = 0; top_four_bin < mz_to_topfour.size(); top_four_bin ++) {
                vector<int> scan_numbers_in_topfour_bin; 
                if (pepmass_bin > 0) {
                    mz_to_topfour_t& mz_to_topfour_prev = pepmass_spectra[pepmass_bin - 1];
                    if (top_four_bin < mz_to_topfour_prev.size()){
                        scan_numbers_in_topfour_bin.insert(scan_numbers_in_topfour_bin.end(), mz_to_topfour_prev[top_four_bin].begin(), mz_to_topfour_prev[top_four_bin].end());
                    }
                } 
                scan_numbers_in_topfour_bin.insert(scan_numbers_in_topfour_bin.end(), mz_to_topfour[top_four_bin].begin(), mz_to_topfour[top_four_bin].end());
                //brute_force_clustering(spectrainfo_all, scan_numbers_in_topfour_bin, cluster_info);
                if ((INDEXING_CHOICE == true) && (scan_numbers_in_topfour_bin.size() > THRESHOLD_INDEXING)) {
                    cluster_local_bin(spectrainfo_all, scan_numbers_in_topfour_bin, cluster_info);
                } else {
                    brute_force_clustering(spectrainfo_all, scan_numbers_in_topfour_bin, cluster_info);
                }
                
            }
            auto topfour_end = chrono::steady_clock::now();
            chrono::duration<double> topfour_pepmass_bin_time = topfour_end - topfour_start;
            pepmassbin_abovethresh_total += float(topfour_pepmass_bin_time.count());
            bins_with_topfour += 1;
        } else {
            vector<int> scan_numbers_in_pepmassbin;
            if (pepmass_bin > 0) {
                vector<int> &prev_scan_numbers_in_pepmassbin = pepmass_distribution[pepmass_bin - 1];
                scan_numbers_in_pepmassbin.insert(scan_numbers_in_pepmassbin.end(), prev_scan_numbers_in_pepmassbin.begin(), prev_scan_numbers_in_pepmassbin.end());
            }
            vector<int> &current_scan_numbers_in_pepmassbin = pepmass_distribution[pepmass_bin];
            scan_numbers_in_pepmassbin.insert(scan_numbers_in_pepmassbin.end(), current_scan_numbers_in_pepmassbin.begin(), current_scan_numbers_in_pepmassbin.end());

            if ((INDEXING_CHOICE == true) && (scan_numbers_in_pepmassbin.size() > THRESHOLD_INDEXING)) {
                brute_force_clustering(spectrainfo_all, scan_numbers_in_pepmassbin, cluster_info);
            } else {
                cluster_local_bin(spectrainfo_all, scan_numbers_in_pepmassbin, cluster_info);
            }
        }
    }
    cout << "the total time needed for top four methods on pepmass bins above the threshold: " << pepmassbin_abovethresh_total << endl;
    cout << "the total number of pepmass bins with topfour applied: "<<  bins_with_topfour << endl;
}

void write_cluster_centers(spectratoinfo_t& spectrainfo_all, vector<int>& scans, string cluster_center_res) {
    spectratomz_t &spectratomz_shared= spectrainfo_all.first;
    spectra_general_info_t &spectra_general_info= spectrainfo_all.second;
    ofstream cluster_center_writer;
    cluster_center_writer.open(cluster_center_res);
    for (int i=0; i < scans.size(); i++) {
        int center_scan = scans[i];
        vector<PeakInfo>& cluster_peaks = spectratomz_shared[center_scan];
        double pepmass = spectra_general_info[center_scan].pepmass;
        double RT = spectra_general_info[center_scan].RT;
        long double magnitude = spectra_general_info[center_scan].magnitude;
        cluster_center_writer << "BEGIN IONS"<< endl;
        cluster_center_writer << "PEPMASS=" << pepmass << endl;
        cluster_center_writer << "RTINSECONDS=" << RT << endl;
        for (int j = 0; j < cluster_peaks.size(); j++) {
            double true_mz = cluster_peaks[j].true_mz;
            double mass = cluster_peaks[j].mass;
            double true_mass = (mass * magnitude) * (mass * magnitude);
            cluster_center_writer << true_mz << " " << true_mass << endl;
        }
        cluster_center_writer << "END IONS" << endl;

    }
    cluster_center_writer.close();
}


int main(int argc, char* argv[]) {
    string input_type = "";
    string output_type = "";
    string input_file = "";
    string output_file = "";
    string arguments = "";
    string cluster_info_output = "cluster_info.csv";
    string printed_cluster_centers = "centers.mgf"; 
    for (int k = 0; k < argc; k++) {
        arguments.append(argv[k]);
        arguments.append(" ");
    }

    cout << arguments << endl;
    bool single_file = true;
    if (argc < 5) {
        cout << "instructions for running the code: " << endl;
        cout << "usage: ./clustering [-i/-l <input_path>] -o <output_path> [--no_topfour] [--bruteforce] [--topfour_threshold <default=5000>] --topfour_resolution <default=0.1> --pepmass_resolution <default=1.0> --peak_resolution <default=0.01> --product_threshold <default = 0.7>" << endl;
        cout << "\n"<< endl;
        cout << "--help: print instructions"<< endl;
        cout << "-i <input_path> : cluster spectra from a single input file " << endl; 
        cout << "-l <input_path> : cluster spectra from a list of input files" << endl; 
        cout << "-o <output_path>: output path for clustering results" << endl;
        cout << "-t <default=0.7> (optional): the minimum threshold of dot product between a cluster member and its corresponding cluster center"<< endl;
        cout << "-c <default=centers.mgf> (optional): the cluster centers produced by the code" << endl;
        cout << "-f X<default=50.0> Y<default=5> (optional): add filtering to peaks in the input spectra. X: the range of filtering peaks; Y: preserving the top Y peaks in each mz range X during filtering" << endl;
        cout << "-s <default=cluster_info.csv>: the information each cluster contains" << endl;
        cout << "--no_topfour (optional): don't apply top four peaks in clustering" << endl;
        cout << "--bruteforce (optional): use bruteforce technique rather than indexing in clustering" << endl;
        cout << "--topfour_threshold <default=5000> (optional): the minimum number of spectra in a pepmass bin to trigger top four peaks filtering. Can't coexist with flag --no_topfour" << endl;
        cout << "--topfour_resolution <default=0.1> (optional): the mz tolerance of two top four peaks. Can't coexist with flag --no_topfour" << endl;
        cout << "--pepmass_resolution <default=1.0> (optional): the minimum pepmass difference to perform dot product between two spectra"<< endl;
        cout << "--peak_resolution <default=0.01> (optional): the minimum mz difference for adding the product between two peaks in calculating the spectra pairwise dotproduct"<< endl;
        cout << "--cluster_minsize <default=2>: the minimum cluster size for a cluster center to be preserved for spectra networking"<< endl; 
        cout << "\n" << endl;
        return 0; 
    }
    for (int i = 1; i < argc; i++) {
        //cout << argc << " " << argv[i] << endl;
        if ((i == argc - 1) && (i==1)) {
            //cout << argc << " " << argv[i] << endl;
            if (strcmp(argv[i], "--help") == 0) {
                cout << "instructions for running the code: " << endl;
                cout << "usage: ./clustering [-i/-l <input_path>] -o <output_path> [--no_topfour] [--bruteforce] [--topfour_threshold <default=5000>] --topfour_resolution <default=0.1> --pepmass_resolution <default=1.0> --peak_resolution <default=0.01> --product_threshold <default = 0.7>" << endl;
                cout << "\n"<< endl;
                cout << "--help: print instructions"<< endl;
                cout << "-i <input_path> : cluster spectra from a single input file " << endl; 
                cout << "-l <input_path> : cluster spectra from a list of input files" << endl; 
                cout << "-o <output_path>: output path for clustering results" << endl;
                cout << "-t <default=0.7> (optional): the minimum threshold of dot product between a cluster member and its corresponding cluster center"<< endl;
                cout << "-c <default=centers.mgf> (optional): the cluster centers produced by the code" << endl;
                cout << "-f X<default=50.0> Y<default=5> (optional): add filtering to peaks in the input spectra. X: the range of filtering peaks; Y: preserving the top Y peaks in each mz range X during filtering" << endl;
                cout << "-s <default=cluster_info.csv>: the information each cluster contains" << endl; 
                cout << "--no_topfour (optional): don't apply top four peaks in clustering" << endl;
                cout << "--bruteforce (optional): use bruteforce technique rather than indexing in clustering" << endl;
                cout << "--topfour_threshold <default=5000> (optional): the minimum number of spectra in a pepmass bin to trigger top four peaks filtering. Can't coexist with flag --no_topfour" << endl;
                cout << "--topfour_resolution <default=0.1> (optional): the mz tolerance of two top four peaks. Can't coexist with flag --no_topfour" << endl;
                cout << "--pepmass_resolution <default=1.0> (optional): the minimum pepmass difference to perform dot product between two spectra"<< endl;
                cout << "--peak_resolution <default=0.01> (optional): the minimum mz difference for adding the product between two peaks in calculating the spectra pairwise dotproduct"<< endl;
                cout << "--cluster_minsize <default=2>: the minimum cluster size for a cluster center to be preserved for spectra networking"<< endl; 
                cout << "\n" << endl;
                return 0; 
            }
        }

        if (strcmp(argv[i], "--no_topfour") == 0) {
            TOPFOUR_CHOICE = false;
        } else if (strcmp(argv[i], "--bruteforce")) {
            INDEXING_CHOICE = false;
        }

        if (i + 1 != argc) {
            if ((strcmp(argv[i], "-i" ) == 0) && (input_type == "")) {                 
                input_type = "-i";    // The next value in the array is your value
                input_file = argv[i+1];
                single_file = true;
                i++;    // Move to the next flag
            } else if ((strcmp(argv[i], "-l" ) == 0) && (input_type == "")) {
                input_type = "-l";
                input_file = argv[i+1];
                single_file = false;
                i++;
            } else if ((strcmp(argv[i], "-o") == 0)) {
                output_file = argv[i+1];
                i++;
            } else if ((strcmp(argv[i], "--topfour_threshold") == 0)) {
                if (TOPFOUR_CHOICE == false) {
                    cout << argv[i] <<" " << argv[i+1] <<": unable to use top-four-peak filtering" << endl;
                    cout << "flag --no_topfour can't coexist with flag --topfour_threshold" << endl;
                    return 1;
                }
                THRESHOLD_TOPFOUR = stoi(argv[i+1]);
                i++;
            } else if (strcmp(argv[i], "--topfour_resolution") == 0) {
                if (TOPFOUR_CHOICE == false) {
                    cout << argv[i] <<" " << argv[i+1] <<": unable to use top-four-peak filtering" << endl;
                    cout << "flag --no_topfour can't coexist with flag --topfour_threshold" << endl;
                    return 1;
                }
                TOPFOURTOLERANCE = stod(argv[i+1]);
                i++;
            } else if (strcmp(argv[i], "--pepmass_resolution") == 0) {
                MASSTOLERANCE = stod(argv[i+1]);
                i++;
            } else if (strcmp(argv[i], "--peak_resolution") == 0) {
                TOLERANCE = stod(argv[i+1]);
                i++;
            } else if (strcmp(argv[i], "-t") == 0) {
                THRESHOLD = stod(argv[i+1]);
                i++;
            } else if (strcmp(argv[i], "-c") == 0) {
                printed_cluster_centers = argv[i+1];
                i++;
            } else if (strcmp(argv[i], "--cluster_minsize") == 0) {
                Output_cluster_minsize = stoi(argv[i+1]);
                i++;
            } else if (strcmp(argv[i], "-s") == 0) {
                cluster_info_output =  argv[i+1];
                i++;
            } else if (i + 2 < argc) {
                if (strcmp(argv[i], "-f") == 0) {
                    SELECTIONRANGE = stod(argv[i+1]);
                    FILTERK = stod(argv[i+2]); 
                    i++;
                    i++;
                } 
            }
        } 
    }

    if (input_file == "") {
        cout << "missing argument, use -i <input_file> or -l <input_file> to specify the path to spectras to cluster"<< endl;
        return 1;
    }

    if (output_file == "") {
        cout << "missing argument, use -o <output_file> to specify the path of the output_file"<< endl;
        return 1;
    }

    if (TOPFOUR_CHOICE == false) {
        THRESHOLD_TOPFOUR = INT_MAX;
    }

    vector<string> filenames;
    if (single_file) {
        filenames.push_back(input_file);
        cout << "input from single file " << endl;
    } else {
        filenames = get_files(input_file);
        cout << "input from file list " << endl;
    }
    
    
    vector<int> scan_centers;
    int parsed_file = 0;
    int start_scan = 0;
    int file_idx_tracer = 0;

    //initialize the cluster center
    auto cluster_start = chrono::steady_clock::now();
    //cout << "tag1" << endl;
    auto preprocessing_start = chrono::steady_clock::now();
    pair<file_info_t ,pair<pepmass_distribution_t, topfour_pepmass_raw_t>> parsed_input = parse_inputs(filenames);
    cout << "tag2" << endl;

    auto preprocessing_end = chrono::steady_clock::now();
    chrono::duration<double> preprocessing_time = preprocessing_end - preprocessing_start;
    cout << "preprocessing_time: " << preprocessing_time.count() << endl;
    //cout << "tag3" << endl;

    auto generating_clusters_start = chrono::steady_clock::now();

    file_info_t &file_info = parsed_input.first;
    unordered_map<string, int> &file_start_scan = file_info.first;
    unordered_map<int, string> &scan_file_src = file_info.second;
    //cout << "tag4" << endl;    

    pepmass_distribution_t &pepmass_distribution = parsed_input.second.first;
    topfour_pepmass_raw_t &topfour_pepmass_raw = parsed_input.second.second;
    cluster_info_t cluster_info;
    spectratoinfo_t& spectrainfo_all = topfour_pepmass_raw.first;
    top_four_combined_t& top_four_combined = topfour_pepmass_raw.second;
    //cout << "tag5" << endl;
    generate_clusters(pepmass_distribution, spectrainfo_all, top_four_combined, cluster_info);
    //cout << "tag6" << endl;
    auto generating_clusters_end = chrono::steady_clock::now();
    chrono::duration<double> generating_clusters_time = generating_clusters_end - generating_clusters_start;
    cout << "time for generating clusters " << generating_clusters_time.count() << endl;
    cout << "total number of spectra: " <<  cluster_info.first.size() << endl;
    cout << "total number of clusters: " << cluster_info.second.size() << endl;

    auto cluster_end = chrono::steady_clock::now();
    chrono::duration<double> total_cluster_time = cluster_end - cluster_start;
    cout << "total clustering time: " << total_cluster_time.count() << endl;

    auto writing_begin = chrono::steady_clock::now();
    ofstream outf;
    outf.open(output_file);

    ofstream cluster_outputf;
    cluster_outputf.open(cluster_info_output);
    
    cluster_outputf << "cluster_idx" << "\t" << "average mz"<< "\t" << "average RT" << "\t"  << "num spectra"  << endl;
    outf << "cluster_idx" << "\t" << "scan"<< "\t" << "mz" << "\t" << "RTINSECONDS" << "\t" << "index in file" << "\t" <<"source filename" << endl;
    
    scan_to_clustercenter_t& spectra_cluster = cluster_info.first;
    clustercenter_to_scans_t& cluster_content = cluster_info.second;
    spectra_general_info_t &spectra_general = spectrainfo_all.second;

    cout << "Cluster Content Size:" << cluster_content.size() << endl;

    for (int cluster_idx = 0; cluster_idx < cluster_content.size(); cluster_idx ++) {
        vector<int> content_tmp = cluster_content[cluster_idx];
        if (content_tmp.size() < Output_cluster_minsize) {
            continue;
        }
        double pepmass_sum = 0.0;
        double RT_sum = 0.0;
        double intensity_sum ;
        int num_spectra = content_tmp.size();
        int connected_component_index;

        for (int content_idx = 0; content_idx < content_tmp.size(); content_idx ++) {
            int query_scan = content_tmp[content_idx];
            if ((content_idx == 0) && (content_tmp.size() >= Output_cluster_minsize))  {
                scan_centers.push_back(query_scan);
            }
            double mz = spectra_general[query_scan].pepmass;
            double RT = spectra_general[query_scan].RT;

            pepmass_sum += mz;
            RT_sum += RT;
            
            string src_filename = scan_file_src[query_scan];
            int local_index = query_scan - file_start_scan[src_filename];

            outf << cluster_idx << "\t" << query_scan << "\t" << mz << "\t" << RT << "\t" << local_index << "\t" << src_filename << endl;
        }
        double pepmass_sum_avg = pepmass_sum / num_spectra;
        double RT_avg = RT_sum / num_spectra;
        cluster_outputf << cluster_idx << "\t" << pepmass_sum_avg << "\t" << RT_avg << "\t" << num_spectra <<  endl;

    }

    outf.close();
    cluster_outputf.close();

    write_cluster_centers(spectrainfo_all, scan_centers, printed_cluster_centers);
    auto writing_end = chrono::steady_clock::now();
    chrono::duration<double> writing_duration = writing_end - writing_begin;
    cout << "Output writing time: " << writing_duration.count() << endl;
    return 0;
}