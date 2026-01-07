"""
================================================================================
RECOMMENDATION ENGINE - Generate actionable recommendations
Based on ISPU category and specific conditions
================================================================================
"""

class RecommendationEngine:
    def __init__(self):
        # Base recommendations per category
        self.recommendations = {
            'Baik': {
                'rumah_tangga': [
                    '✅ Buka jendela untuk sirkulasi udara',
                    '🌱 Aktivitas berkebun dapat dilakukan',
                    '🏃 Olahraga outdoor direkomendasikan'
                ],
                'transportasi': [
                    '🚴 Bersepeda atau jalan kaki aman',
                    '🚗 Tidak ada pembatasan kendaraan',
                    '🌳 Gunakan rute dengan banyak pepohonan'
                ],
                'kesehatan': [
                    '😊 Tidak ada risiko khusus',
                    '👶 Anak-anak dapat bermain outdoor',
                    '👴 Lansia dapat beraktivitas normal'
                ],
                'perkantoran': [
                    '🏢 Ventilasi alami dapat digunakan',
                    '💻 Bekerja dari outdoor area aman',
                    '☕ Coffee break outdoor diperbolehkan'
                ],
                'lingkungan': [
                    '🌿 Tanam lebih banyak pohon',
                    '♻️ Lakukan composting organik',
                    '🚯 Kampanye zero waste'
                ],
                'komunitas': [
                    '🎉 Event outdoor dapat diadakan',
                    '🏃‍♀️ Car free day efektif',
                    '📢 Sosialisasi hidup berkelanjutan'
                ]
            },
            'Sedang': {
                'rumah_tangga': [
                    '🪟 Buka jendela saat pagi/sore',
                    '🧹 Bersihkan rumah lebih sering',
                    '🌱 Tanaman indoor membantu filtrasi'
                ],
                'transportasi': [
                    '🚌 Gunakan transportasi umum',
                    '⏰ Hindari jam sibuk jika memungkinkan',
                    '🚗 Carpooling lebih diutamakan'
                ],
                'kesehatan': [
                    '⚠️ Kelompok sensitif perlu waspada',
                    '👶 Batasi waktu bermain anak di luar',
                    '💊 Siapkan obat untuk penyakit pernapasan'
                ],
                'perkantoran': [
                    '❄️ Gunakan AC dengan filter HEPA',
                    '🪴 Tambah tanaman indoor di kantor',
                    '⏰ Atur jadwal istirahat outdoor'
                ],
                'lingkungan': [
                    '🚫 Hindari pembakaran sampah',
                    '💧 Siram jalanan untuk kurangi debu',
                    '🌳 Rawat pohon yang ada'
                ],
                'komunitas': [
                    '📱 Share info kualitas udara',
                    '🚴 Promosi bersepeda ke tempat kerja',
                    '♻️ Program reduce-reuse-recycle'
                ]
            },
            'Tidak Sehat': {
                'rumah_tangga': [
                    '🚪 Tutup jendela dan pintu',
                    '❄️ Gunakan AC dengan filter HEPA',
                    '😷 Sediakan masker untuk keluarga',
                    '🧼 Bersihkan permukaan lebih sering'
                ],
                'transportasi': [
                    '🚗 Tutup jendela mobil, gunakan AC',
                    '🚌 Prioritas transportasi umum',
                    '🏠 Work from home jika memungkinkan',
                    '⏰ HINDARI jam sibuk (07:00-09:00, 16:00-18:00)'
                ],
                'kesehatan': [
                    '😷 Gunakan masker N95/KN95',
                    '🏥 Konsultasi dokter jika bermasalah',
                    '👶 Anak & lansia tetap di dalam',
                    '💊 Siapkan inhaler untuk asma',
                    '💧 Minum lebih banyak air'
                ],
                'perkantoran': [
                    '🏢 Aktifkan air purifier di ruangan',
                    '🚫 Batalkan meeting outdoor',
                    '💻 Izinkan WFH untuk karyawan',
                    '⏰ Kurangi aktivitas lapangan'
                ],
                'lingkungan': [
                    '🚫 STOP pembakaran apapun',
                    '🚗 Kurangi penggunaan kendaraan',
                    '🏭 Monitoring emisi industri',
                    '💦 Water spray untuk kurangi partikel'
                ],
                'komunitas': [
                    '📢 Kampanye stay at home',
                    '🚫 Tunda event outdoor',
                    '🏥 Siaga medis untuk kelompok rentan',
                    '📱 Update real-time kualitas udara'
                ]
            },
            'Sangat Tidak Sehat': {
                'rumah_tangga': [
                    '🚪 TUTUP RAPAT semua jendela',
                    '❄️ AC dengan filter HEPA WAJIB',
                    '😷 Gunakan masker BAHKAN di dalam',
                    '🧽 Pel lantai setiap hari',
                    '🌱 Tanaman indoor untuk O2 tambahan'
                ],
                'transportasi': [
                    '🏠 WAJIB work from home',
                    '🚫 HINDARI semua perjalanan',
                    '🚗 Jika darurat: tutup jendela + AC recirculate',
                    '😷 Selalu gunakan masker N95'
                ],
                'kesehatan': [
                    '🚨 ALERT: Kelompok rentan di rumah!',
                    '😷 Masker N95 WAJIB di luar ruangan',
                    '🏥 Hotline medis siaga',
                    '💊 Stok obat pernapasan mencukupi',
                    '💧 Hidrasi maksimal',
                    '🚫 TIDAK ADA aktivitas fisik berat'
                ],
                'perkantoran': [
                    '🏠 TUTUP kantor, full WFH',
                    '🚫 Semua operasi lapangan dihentikan',
                    '❄️ Gedung: seal ventilasi + air purifier max',
                    '📱 Virtual meeting ONLY'
                ],
                'lingkungan': [
                    '🚨 EMERGENCY: Stop semua emisi',
                    '🏭 Industri: Kurangi produksi',
                    '🚗 Odd-even atau car-free',
                    '💦 Water canon/spray intensif',
                    '📢 Declare emergency status'
                ],
                'komunitas': [
                    '📢 SIAGA DARURAT polusi',
                    '🏥 Posko kesehatan siaga 24 jam',
                    '🚫 SEMUA event outdoor DIBATALKAN',
                    '🆘 Bantuan untuk warga rentan',
                    '📱 Broadcast warning massal'
                ]
            },
            'Berbahaya': {
                'rumah_tangga': [
                    '🚨 LOCKDOWN: Tetap di dalam!',
                    '😷 Masker N95 WAJIB bahkan di dalam',
                    '❄️ Multiple air purifier di setiap ruangan',
                    '🚪 Seal celah pintu/jendela',
                    '🧽 Wet cleaning 2x sehari'
                ],
                'transportasi': [
                    '🚨 TOTAL LOCKDOWN transportasi',
                    '🚫 ZERO perjalanan kecuali darurat',
                    '🏥 Ambulans & emergency only',
                    '😷 Hazmat level protection'
                ],
                'kesehatan': [
                    '🚨 KONDISI DARURAT KESEHATAN!',
                    '🏥 RS siaga penuh 24/7',
                    '😷 N95/P100 respirator WAJIB',
                    '💊 Distribusi obat ke warga',
                    '🚑 Evakuasi kelompok sangat rentan',
                    '☠️ Risiko kematian TINGGI'
                ],
                'perkantoran': [
                    '🚨 FULL SHUTDOWN semua kantor',
                    '🏢 Essential services ONLY',
                    '❄️ Hermetic seal gedung',
                    '😷 Full PPE untuk staff essential'
                ],
                'lingkungan': [
                    '🚨 DEKLARASI BENCANA',
                    '🏭 SHUTDOWN industri non-esensial',
                    '🚗 TOTAL BAN kendaraan pribadi',
                    '💦 Emergency response team',
                    '✈️ Cloud seeding jika memungkinkan'
                ],
                'komunitas': [
                    '🚨 STATUS BENCANA NASIONAL',
                    '🏥 Emergency response center',
                    '🚁 Evakuasi massal siap',
                    '🆘 Distribusi masker & air purifier',
                    '📺 Broadcast 24/7 emergency',
                    '💰 Dana bantuan darurat'
                ]
            }
        }
    
    def get_recommendations(self, ispu_category):
        """
        Get recommendations based on ISPU category
        
        Args:
            ispu_category: str ('Baik', 'Sedang', 'Tidak Sehat', etc.)
            
        Returns:
            dict with recommendations per category
        """
        return self.recommendations.get(ispu_category, self.recommendations['Sedang'])
    
    def get_specific_advice(self, ispu_category, context=None):
        """
        Get specific advice based on context
        
        Args:
            ispu_category: ISPU category
            context: dict with additional context (time, weather, etc.)
            
        Returns:
            list of specific recommendations
        """
        base_recs = self.get_recommendations(ispu_category)
        specific_advice = []
        
        # Time-based recommendations
        if context and 'hour' in context:
            hour = context['hour']
            
            # Rush hour warnings
            if 7 <= hour <= 9 or 16 <= hour <= 18:
                if ispu_category in ['Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']:
                    specific_advice.append('⚠️ JAM SIBUK: Polusi sangat tinggi, HINDARI keluar!')
                elif ispu_category == 'Sedang':
                    specific_advice.append('⚠️ JAM SIBUK: Polusi cenderung lebih tinggi')
            
            # Night time
            if hour >= 22 or hour <= 5:
                specific_advice.append('🌙 Malam hari: Tutup jendela, udara lebih dingin dan stabil')
        
        # Weather-based recommendations
        if context and 'weather' in context:
            if context['weather'] == 'rain':
                specific_advice.append('🌧️ Hujan membantu menurunkan polusi, kualitas udara akan membaik')
            elif context['weather'] == 'windy':
                specific_advice.append('💨 Angin kencang membantu dispersi polutan')
            elif context['weather'] == 'calm':
                if ispu_category in ['Tidak Sehat', 'Sangat Tidak Sehat']:
                    specific_advice.append('⚠️ Tidak ada angin: Polutan terperangkap, extra hati-hati')
        
        # Temperature-based
        if context and 'temperature' in context:
            temp = context['temperature']
            if temp > 30:
                specific_advice.append('🌡️ Suhu tinggi: Pembentukan O3 meningkat, hindari outdoor siang hari')
        
        # Add base recommendations
        for category_recs in base_recs.values():
            specific_advice.extend(category_recs[:2])  # Take top 2 from each
        
        return specific_advice[:10]  # Limit to 10 recommendations
    
    def get_emergency_contacts(self):
        """Get emergency contact information"""
        return {
            'medical': {
                'ambulance': '118',
                'rs_darurat': '119',
                'puskesmas': '021-500-567'
            },
            'environmental': {
                'dlh_malang': '(0341) 551-111',
                'bmkg': '196',
                'damkar': '113'
            },
            'government': {
                'posko_bencana': '021-2987-5300',
                'satpol_pp': '(0341) 551-234'
            }
        }
    
    def get_activity_restrictions(self, ispu_category):
        """
        Get activity restrictions based on ISPU
        
        Returns:
            dict with allowed/restricted activities
        """
        restrictions = {
            'Baik': {
                'allowed': ['Semua aktivitas outdoor', 'Olahraga', 'Berkebun', 'Event outdoor'],
                'restricted': [],
                'prohibited': []
            },
            'Sedang': {
                'allowed': ['Aktivitas ringan outdoor', 'Jalan santai', 'Berkebun pagi/sore'],
                'restricted': ['Olahraga berat outdoor', 'Marathon', 'Bersepeda jarak jauh'],
                'prohibited': []
            },
            'Tidak Sehat': {
                'allowed': ['Aktivitas indoor', 'Olahraga indoor'],
                'restricted': ['Jalan singkat dengan masker', 'Belanja cepat'],
                'prohibited': ['Olahraga outdoor', 'Event outdoor', 'Aktivitas berat']
            },
            'Sangat Tidak Sehat': {
                'allowed': ['Tetap di dalam ruangan'],
                'restricted': ['Perjalanan darurat saja'],
                'prohibited': ['SEMUA aktivitas outdoor', 'Olahraga', 'Kumpul outdoor']
            },
            'Berbahaya': {
                'allowed': [],
                'restricted': [],
                'prohibited': ['SEMUA aktivitas di luar', 'Buka jendela', 'Ventilasi alami']
            }
        }
        
        return restrictions.get(ispu_category, restrictions['Sedang'])