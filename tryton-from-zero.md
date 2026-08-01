# Tryton Dari Nol (Khusus Proyek LWF02-WMS)

Panduan ini dibuat untuk developer yang benar-benar baru di Tryton.

Kalau Anda terbiasa dengan MVC tradisional, baca dokumen ini urut dari atas ke bawah. Setelah selesai, Anda akan paham:
- Tryton itu apa dan cara berpikirnya
- UI/menu itu datang dari mana
- "Routing" di Tryton sebenarnya seperti apa
- `wms`, `file_import`, dan `lotte` perannya apa
- alur kerja development feature sehari-hari

## 1) Tryton Itu Apa (Versi Praktis)

Tryton adalah ERP framework Python yang model-driven.

Di proyek ini, Anda tidak membuat halaman web per URL seperti framework MVC web biasa.
Yang Anda definisikan adalah:
- model Python
- view XML (form/tree)
- action + menu XML
- wizard/report

Lalu Tryton yang merakit semuanya menjadi UI dan API RPC.

## 2) Mapping Cepat Dari MVC ke Tryton

Mapping praktis untuk developer MVC:
- Route URL controller -> RPC method (contoh `model.stock.picking.assign`)
- Controller -> method pada model/wizard
- Template view -> XML view (`form`, `tree`)
- Service layer -> biasanya langsung di method model/wizard (atau helper module)
- Middleware transaction -> ditangani Tryton dispatcher + Transaction

## 3) Struktur Proyek Ini (Yang Wajib Dikenal)

Root penting:
- `/projects/config` -> konfigurasi server (`trytond.conf`, log config)
- `/projects/scripts` -> script operasional (`run.sh`, `install.sh`, `update.sh`, `docs.sh`)
- `/projects/src` -> semua modul custom

Modul di `/projects/src`:
- `wms` -> core logic warehouse (picking, inbound, outbound, opname, print, PDA API)
- `file_import` -> sync file host (receive/send, queue, cron)
- `lotte` -> lapisan custom perusahaan, bergantung pada `wms` + `file_import`

Catatan penting:
- `src/lotte/tryton.cfg` berisi `depends: wms, file_import`
- artinya `lotte` adalah extension/overlay di atas core

## 4) UI Tryton Datangnya Dari Mana?

Ini poin yang paling sering bikin bingung pemula.

UI Tryton TIDAK membaca file `menus.xml` langsung saat browser request.
Alur aslinya:
1. Module di-update (`-u <module>`)
2. Tryton import XML (`menu`, `action`, `view`, `security`) ke database
3. Saat user login, SAO minta menu ke server via RPC
4. Server kirim struktur menu dari database

Jadi file XML adalah "source definition", sementara yang dipakai runtime adalah data yang sudah tersimpan di DB.

## 5) Routing Tryton (Yang Sebenarnya)

Ada 2 layer routing:

1. HTTP route server:
- endpoint utama RPC: `/<database_name>/`
- ditangani oleh Tryton WSGI + dispatcher

2. RPC route aplikasi:
- format method: `model.<model_name>.<method>`
- contoh: `model.stock.picking.assign`
- dispatcher resolve method ini ke object di Pool

File framework penting:
- `trytond/protocols/dispatcher.py` -> route RPC, auth, dispatch, transaction
- `trytond/protocols/jsonrpc.py` -> parse request JSON, encode response JSON

## 6) Flow Nyata: Dari Klik Menu Sampai Data Tampil

Contoh: user buka menu Picking.

1. Definisi menu ada di `src/wms/menus.xml`
2. Menu menunjuk action `wms.act_picking_form`
3. Action/view didefinisikan di `src/wms/picking.xml`
4. SAO kirim JSON-RPC ke `/<database_name>/`
5. Dispatcher resolve method model `stock.picking`
6. Model `stock.picking` ada di `src/wms/picking.py`
7. Hasil query/record dikirim balik JSON ke SAO
8. SAO render list/form ke user

## 7) Flow Nyata: Klik Tombol Workflow

Contoh: tombol Assign di Picking.

1. Tombol didefinisikan di XML (`src/wms/picking.xml`)
2. SAO kirim RPC method button ke server
3. Dispatcher panggil method Python `assign` di `src/wms/picking.py`
4. Method jalan dalam transaction (state berubah, save)
5. Commit
6. Response kembali ke SAO, tampilan state berubah

## 8) Flow Nyata: API Untuk PDA

Contoh: endpoint custom `wms.api`.

1. RPC method custom didaftarkan di `src/wms/api.py` pada `__rpc__`
2. PDA kirim JSON-RPC ke endpoint `/<database_name>/`
3. Dispatcher resolve ke model `wms.api`
4. Method Python jalan, baca/tulis model lain via Pool
5. Response JSON dikirim balik ke client PDA

## 9) Kenapa Menu Kadang Tidak Muncul?

Checklist paling umum:
1. Modul belum di-update setelah ubah XML
2. `menuitem` parent salah (nyangkut di cabang menu lain)
3. `active="0"` (menu sengaja disembunyikan)
4. user tidak punya akses group/menu
5. action/view yang dirujuk belum ada atau belum terimport

## 10) Siklus Development Harian (Simple)

Urutan kerja yang aman:
1. ambil tiket
2. tentukan modul target (`wms` vs `lotte` vs `file_import`)
3. ubah Python + XML + security
4. update modul ke DB
5. test manual di UI
6. test otomatis
7. PR ke `develop`

Contoh command:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/LWF02-xxx-feature-name

bash /projects/scripts/install.sh

trytond-admin \
  -c /projects/config/trytond.conf \
  --logconf /projects/config/trytond.log.conf \
  -d wms \
  --activate-dependencies \
  -u lotte

python -m pytest /projects/src/lotte/tests/
```

## 11) Membuat Feature Baru (Template Pemula)

Template pikir saat bikin fitur baru:

1. Tentukan kebutuhan
- contoh: "tambah status prioritas di Picking"

2. Tentukan lokasi code
- generic warehouse: `wms`
- company-specific lotte: `lotte`

3. Implement backend
- tambah field/method model di Python
- kalau perlu wizard baru

4. Implement UI XML
- tambah field di form/tree view
- tambah button/action/menu bila perlu

5. Implement security
- model access
- button/menu access

6. Daftarkan di `tryton.cfg`
- file XML masuk daftar `xml:`
- model/wizard/report masuk `[register]`

7. Update module + test

## 12) Starting Point Saat Anda Bingung Harus Baca Dari Mana

Urutan investigasi paling efektif:
1. lihat `tryton.cfg` modul
2. cari `menu.xml` / `menus.xml`
3. cari action id di file XML domain terkait (misalnya `picking.xml`)
4. lihat `res_model` atau `wiz_name`
5. buka file Python model/wizard yang sesuai
6. cari method button/workflow yang dipanggil

## 13) Diagram Ringkas Arsitektur Proyek Ini

```{mermaid}
flowchart TD
    A[User SAO / PDA] --> B[Tryton WSGI]
    B --> C[RPC Dispatcher]
    C --> D[wms module]
    C --> E[file_import module]
    C --> F[lotte module]
    D --> G[(PostgreSQL)]
    E --> G
    F --> G
```

## 14) Quick Recap

- Tryton bukan web MVC route-per-page.
- UI dibentuk dari metadata XML yang diimport ke DB.
- Runtime call utama lewat RPC endpoint `/<database_name>/`.
- Logic utama proyek ini ada di `wms`, `lotte` sebagai layer custom.
- Untuk pemula: selalu trace dari `menu/action/view` -> `model/wizard method` -> DB.

## 15) Setelah Paham Dasar, Lanjut Ke Mana?

Rekomendasi urutan belajar di repo ini:
1. `src/wms/menus.xml`
2. `src/wms/picking.xml`
3. `src/wms/picking.py`
4. `src/wms/api.py`
5. `src/file_import/sync_file.py`
6. baru ke extension di `src/lotte`
