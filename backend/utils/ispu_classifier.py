"""
================================================================================
ISPU CLASSIFIER - Classify air quality based on Indonesian standards
Based on: KEP-45/MENLH/10/1997 and Peraturan Menteri LH No. 14/2010
================================================================================
"""

class ISPUClassifier:
    def __init__(self):
        # ISPU Breakpoints for PM2.5 (μg/m³)
        self.pm25_breakpoints = [
            (0, 15.5, 'Baik'),
            (15.6, 55.4, 'Sedang'),
            (55.5, 150.4, 'Tidak Sehat'),
            (150.5, 250.4, 'Sangat Tidak Sehat'),
            (250.5, float('inf'), 'Berbahaya')
        ]
        
        # ISPU Breakpoints for O3 (μg/m³) - 8 hour average
        self.o3_breakpoints = [
            (0, 120, 'Baik'),
            (121, 235, 'Sedang'),
            (236, 400, 'Tidak Sehat'),
            (401, 800, 'Sangat Tidak Sehat'),
            (801, float('inf'), 'Berbahaya')
        ]
        
        # ISPU Breakpoints for CO (μg/m³) - 8 hour average
        self.co_breakpoints = [
            (0, 4000, 'Baik'),
            (4001, 8000, 'Sedang'),
            (8001, 15000, 'Tidak Sehat'),
            (15001, 30000, 'Sangat Tidak Sehat'),
            (30001, float('inf'), 'Berbahaya')
        ]
        
        # Category colors for frontend
        self.category_colors = {
            'Baik': '#22c55e',           # green-500
            'Sedang': '#eab308',         # yellow-500
            'Tidak Sehat': '#f97316',    # orange-500
            'Sangat Tidak Sehat': '#ef4444',  # red-500
            'Berbahaya': '#7f1d1d'       # red-900
        }
        
        # Category priorities (for determining overall status)
        self.category_priority = {
            'Baik': 1,
            'Sedang': 2,
            'Tidak Sehat': 3,
            'Sangat Tidak Sehat': 4,
            'Berbahaya': 5
        }
        
        # Health recommendations per category
        self.health_advice = {
            'Baik': 'Kualitas udara sangat baik untuk aktivitas luar ruangan',
            'Sedang': 'Kualitas udara dapat diterima untuk sebagian besar orang',
            'Tidak Sehat': 'Mulai berdampak pada kelompok sensitif',
            'Sangat Tidak Sehat': 'Berbahaya bagi semua kelompok, terutama sensitif',
            'Berbahaya': 'Sangat berbahaya, hindari aktivitas luar ruangan'
        }
    
    def classify_pm25(self, value):
        """
        Classify PM2.5 concentration
        
        Args:
            value: PM2.5 concentration in μg/m³
            
        Returns:
            str: ISPU category
        """
        return self._classify(value, self.pm25_breakpoints)
    
    def classify_o3(self, value):
        """
        Classify O3 concentration
        
        Args:
            value: O3 concentration in μg/m³
            
        Returns:
            str: ISPU category
        """
        return self._classify(value, self.o3_breakpoints)
    
    def classify_co(self, value):
        """
        Classify CO concentration
        
        Args:
            value: CO concentration in μg/m³
            
        Returns:
            str: ISPU category
        """
        return self._classify(value, self.co_breakpoints)
    
    def _classify(self, value, breakpoints):
        """
        Generic classification based on breakpoints
        
        Args:
            value: pollutant concentration
            breakpoints: list of (min, max, category) tuples
            
        Returns:
            str: ISPU category
        """
        for min_val, max_val, category in breakpoints:
            if min_val <= value <= max_val:
                return category
        
        # If value exceeds all ranges, return worst category
        return breakpoints[-1][2]
    
    def classify_all(self, pm25, o3, co):
        """
        Classify all pollutants and determine overall status
        
        Args:
            pm25: PM2.5 concentration in μg/m³
            o3: O3 concentration in μg/m³
            co: CO concentration in μg/m³
            
        Returns:
            dict with individual and overall classifications
        """
        pm25_cat = self.classify_pm25(pm25)
        o3_cat = self.classify_o3(o3)
        co_cat = self.classify_co(co)
        
        # Overall status is the worst (highest priority) category
        categories = [pm25_cat, o3_cat, co_cat]
        overall = max(categories, key=lambda x: self.category_priority[x])
        
        return {
            'pm25': pm25_cat,
            'o3': o3_cat,
            'co': co_cat,
            'overall': overall,
            'color': self.category_colors[overall],
            'advice': self.health_advice[overall]
        }
    
    def get_ispu_value(self, pollutant, concentration):
        """
        Calculate ISPU index value (0-500+)
        
        Args:
            pollutant: 'pm25', 'o3', or 'co'
            concentration: pollutant concentration
            
        Returns:
            int: ISPU index value
        """
        if pollutant == 'pm25':
            breakpoints = self.pm25_breakpoints
        elif pollutant == 'o3':
            breakpoints = self.o3_breakpoints
        elif pollutant == 'co':
            breakpoints = self.co_breakpoints
        else:
            return 0
        
        # ISPU index ranges
        ispu_ranges = [
            (0, 50),      # Baik
            (51, 100),    # Sedang
            (101, 199),   # Tidak Sehat
            (200, 299),   # Sangat Tidak Sehat
            (300, 500)    # Berbahaya
        ]
        
        # Find matching breakpoint
        for i, (min_conc, max_conc, category) in enumerate(breakpoints):
            if min_conc <= concentration <= max_conc:
                # Linear interpolation within range
                ispu_min, ispu_max = ispu_ranges[i]
                
                if max_conc == float('inf'):
                    return ispu_max
                
                # Calculate ISPU value
                ispu = ispu_min + (ispu_max - ispu_min) * (concentration - min_conc) / (max_conc - min_conc)
                return int(ispu)
        
        # If exceeds all ranges
        return 500
    
    def get_detailed_info(self, category):
        """
        Get detailed information about ISPU category
        
        Args:
            category: ISPU category string
            
        Returns:
            dict with detailed information
        """
        details = {
            'Baik': {
                'range': '0-50',
                'description': 'Tingkat kualitas udara yang tidak memberikan efek bagi kesehatan manusia atau hewan dan tidak berpengaruh pada tumbuhan, bangunan ataupun nilai estetika',
                'health_impact': 'Tidak ada',
                'sensitive_groups': [],
                'activity_advice': 'Aktivitas normal'
            },
            'Sedang': {
                'range': '51-100',
                'description': 'Tingkat kualitas udara yang tidak berpengaruh pada kesehatan manusia ataupun hewan tetapi berpengaruh pada tumbuhan yang sensitif dan nilai estetika',
                'health_impact': 'Minimal pada kelompok sangat sensitif',
                'sensitive_groups': ['Orang dengan penyakit paru-paru yang parah'],
                'activity_advice': 'Aktivitas normal, kelompok sensitif perlu waspada'
            },
            'Tidak Sehat': {
                'range': '101-199',
                'description': 'Tingkat kualitas udara yang bersifat merugikan pada manusia ataupun kelompok hewan yang sensitif atau bisa menimbulkan kerusakan pada tumbuhan',
                'health_impact': 'Meningkatkan risiko kesehatan pada kelompok sensitif',
                'sensitive_groups': [
                    'Anak-anak',
                    'Lansia',
                    'Penderita penyakit pernapasan',
                    'Penderita penyakit jantung'
                ],
                'activity_advice': 'Kurangi aktivitas luar ruangan yang berat dan lama'
            },
            'Sangat Tidak Sehat': {
                'range': '200-299',
                'description': 'Tingkat kualitas udara yang dapat merugikan kesehatan pada sejumlah segmen populasi yang terpapar',
                'health_impact': 'Meningkatkan risiko kesehatan pada semua orang',
                'sensitive_groups': [
                    'Semua kelompok masyarakat',
                    'Sangat berbahaya bagi kelompok sensitif'
                ],
                'activity_advice': 'Hindari aktivitas luar ruangan'
            },
            'Berbahaya': {
                'range': '300+',
                'description': 'Tingkat kualitas udara berbahaya yang secara umum dapat merugikan kesehatan yang serius pada populasi',
                'health_impact': 'Serius pada semua orang',
                'sensitive_groups': ['Semua kelompok masyarakat'],
                'activity_advice': 'Tetap di dalam ruangan dan tutup ventilasi'
            }
        }
        
        return details.get(category, {})
    
    def get_color_for_value(self, pollutant, value):
        """
        Get color code for a specific pollutant value
        
        Args:
            pollutant: 'pm25', 'o3', or 'co'
            value: concentration value
            
        Returns:
            str: hex color code
        """
        if pollutant == 'pm25':
            category = self.classify_pm25(value)
        elif pollutant == 'o3':
            category = self.classify_o3(value)
        elif pollutant == 'co':
            category = self.classify_co(value)
        else:
            return '#9ca3af'  # gray-400 for unknown
        
        return self.category_colors[category]
    
    def is_safe_for_outdoor(self, pm25, o3, co):
        """
        Check if air quality is safe for outdoor activities
        
        Args:
            pm25, o3, co: pollutant concentrations
            
        Returns:
            dict with safety status and recommendations
        """
        classification = self.classify_all(pm25, o3, co)
        overall = classification['overall']
        
        is_safe = overall in ['Baik', 'Sedang']
        
        result = {
            'is_safe': is_safe,
            'category': overall,
            'color': classification['color'],
            'recommendation': ''
        }
        
        if overall == 'Baik':
            result['recommendation'] = '✅ Aman untuk semua aktivitas outdoor'
        elif overall == 'Sedang':
            result['recommendation'] = '⚠️ Aman untuk aktivitas normal, kelompok sensitif perlu waspada'
        elif overall == 'Tidak Sehat':
            result['recommendation'] = '⛔ Kurangi aktivitas outdoor yang berat'
        elif overall == 'Sangat Tidak Sehat':
            result['recommendation'] = '🚫 Hindari aktivitas outdoor'
        else:  # Berbahaya
            result['recommendation'] = '🔴 BAHAYA! Tetap di dalam ruangan'
        
        return result
    
    def get_mask_recommendation(self, pm25):
        """
        Recommend mask type based on PM2.5 level
        
        Args:
            pm25: PM2.5 concentration
            
        Returns:
            str: mask recommendation
        """
        if pm25 <= 15.5:
            return 'Tidak perlu masker'
        elif pm25 <= 55.4:
            return 'Masker kain/bedah (opsional)'
        elif pm25 <= 150.4:
            return 'Masker N95/KN95 direkomendasikan'
        else:
            return 'Masker N95/KN95 WAJIB digunakan'