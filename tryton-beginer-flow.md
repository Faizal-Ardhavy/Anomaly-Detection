# Tryton Beginner Flow (LWF02-WMS)

Panduan ini dibuat untuk developer yang terbiasa dengan MVC tradisional dan baru masuk ke Tryton.

Tujuan panduan:
- Paham alur saat user membuka UI sampai method Python yang dieksekusi.
- Paham struktur folder proyek ini (`wms`, `file_import`, `lotte`).
- Paham alur bikin feature baru secara praktis.

## 1. Mental Model: MVC vs Tryton

Kalau di MVC tradisional, biasanya seperti ini:
- Route -> Controller -> Service -> Model -> DB -> View

Di Tryton (proyek ini), pendekatan praktisnya:
- Menu/Action UI -> RPC method (`model.<name>.<method>`) -> Model/Wizard/Report -> DB

Jadi "controller" di Tryton umumnya tersebar di:
- Method model (`ModelSQL`/`ModelView`)
- Wizard (`Wizard` states)
- Report

## 2. Struktur Proyek yang Paling Penting

Folder utama:
- `/projects/config`: konfigurasi server Tryton dan logging.
- `/projects/scripts`: script operasional (`run`, `install`, `update`, `docs`).
- `/projects/src`: modul custom.

Modul custom di `/projects/src`:
- `wms`: core bisnis warehouse (picking, inbound, outbound, opname, print, PDA API).
- `file_import`: sinkronisasi file (receive/send, cron/queue).
- `lotte`: lapisan custom company-specific (bergantung pada `wms` dan `file_import`).

Dependency modul terlihat di `src/lotte/tryton.cfg`:
- `depends: wms, file_import`

Artinya `lotte` dipasang di atas modul core.

## 3. Alur Request: Dari UI ke Python

Contoh saat user klik menu Picking di web UI:

```{mermaid}
flowchart LR
    A[Browser SAO] --> B[Tryton WSGI]
    B --> C[JSON-RPC Endpoint /<db>/]
    C --> D[Dispatcher: resolve method RPC]
    D --> E[Model/Wizard Method]
    E --> F[(PostgreSQL)]
    F --> E
    E --> G[JSON Response]
    G --> A
```

Komponen penting:
- Endpoint RPC ditangani oleh dispatcher Tryton di `trytond/protocols/dispatcher.py`.
- Protokol JSON ditangani di `trytond/protocols/jsonrpc.py`.
- Method dieksekusi dalam transaction DB, commit/rollback otomatis.

## 4. UI Dipanggil dari Mana?

Di Tryton, UI bukan React/Vue page per route. UI dibentuk oleh metadata XML.

Urutan hubungan:
1. `menu.xml` mendefinisikan menu.
2. Menu menunjuk ke action (`ir.action.act_window` / wizard action).
3. Action menunjuk ke model (`res_model`) dan view (tree/form).
4. Saat user klik tombol form, Tryton memanggil method Python yang didaftarkan sebagai button/wizard.

Contoh real di repo ini:
- Menu operations/picking: `src/wms/menus.xml`
- Action + view picking: `src/wms/picking.xml`
- Logic tombol/workflow picking: `src/wms/picking.py`

## 5. "Routing" dalam Konteks Tryton

Ada dua jenis routing yang perlu dipahami:

1. HTTP route (level server):
- Ditangani oleh WSGI app Tryton.
- Contoh endpoint penting: `/<database_name>/` untuk RPC.

2. RPC route (level aplikasi):
- Nama method semacam `model.stock.picking.assign`.
- Dispatcher memetakan string method ini ke object model/wizard/report di Pool.

Jadi kalau Anda cari "controller route", jejaknya biasanya:
- XML action/menu -> nama model/wizard
- Python class model/wizard -> method yang dipanggil

## 6. Contoh Flow File-per-File (Nyata di Repo Ini)

Bagian ini menjawab pertanyaan: "akses route apa, awalnya file mana, lanjut ke mana, sampai data tampil ke user lewat file apa".

### Contoh A: User buka menu Picking (list data)

1. Menu diklik dari UI berdasarkan definisi di `src/wms/menus.xml`:
  - menu `menu_wh_picking` menunjuk action `wms.act_picking_form`.
2. Action `wms.act_picking_form` didefinisikan di `src/wms/picking.xml`:
  - `res_model = stock.picking`
  - view list/form untuk model `stock.picking`.
3. Browser SAO melakukan JSON-RPC ke endpoint `/<database_name>/`.
4. Endpoint RPC ditangani Tryton dispatcher di `trytond/protocols/dispatcher.py` (`rpc` -> `_dispatch`).
5. Dispatcher resolve method model (`stock.picking`) via Pool, lalu jalankan method read/search bawaan model.
6. Struktur field model yang dibaca berasal dari class `Picking` di `src/wms/picking.py`.
7. Hasil diserialisasi sebagai JSON oleh `trytond/protocols/jsonrpc.py`, lalu dikirim balik ke browser dan dirender sebagai list/form.

Ringkasnya:
`src/wms/menus.xml` -> `src/wms/picking.xml` -> RPC `/<db>/` -> `trytond/protocols/dispatcher.py` -> `src/wms/picking.py` -> `trytond/protocols/jsonrpc.py` -> UI.

### Contoh B: User klik tombol Assign pada Picking

1. Tombol didefinisikan di `src/wms/picking.xml`:
  - `ir.model.button` name `assign` untuk model `stock.picking`.
2. Saat user klik tombol, SAO kirim RPC method model ke endpoint `/<database_name>/`.
3. Dispatcher di `trytond/protocols/dispatcher.py` memvalidasi RPC method dan hak akses, lalu memanggil method Python.
4. Method yang dipanggil adalah `assign` pada class `Picking` di `src/wms/picking.py`.
5. Di method itu, workflow transition dijalankan (`assigned`) dan data disimpan ke DB (commit transaction).
6. Response JSON dikirim balik ke UI; status record di layar berubah sesuai state terbaru.

Ringkasnya:
Button XML (`src/wms/picking.xml`) -> RPC -> method `Picking.assign` (`src/wms/picking.py`) -> commit DB -> JSON response -> UI refresh.

### Contoh C: Client PDA panggil API custom `wms.api`

1. Method API custom didaftarkan di `src/wms/api.py` pada `__rpc__` (misalnya `get_putto_load`, `find`, `modify`).
2. Client PDA kirim JSON-RPC ke endpoint yang sama: `/<database_name>/`.
3. Dispatcher di `trytond/protocols/dispatcher.py` resolve method ke model `wms.api`.
4. Method di `src/wms/api.py` dieksekusi, misalnya:
  - ambil model lain via `Pool().get(...)`
  - query data
  - return payload/list id.
5. `trytond/protocols/jsonrpc.py` membungkus hasil sebagai JSON response.
6. PDA menerima response dan menampilkan data ke user mobile.

Ringkasnya:
PDA request -> `trytond/protocols/dispatcher.py` -> `src/wms/api.py` -> DB -> `trytond/protocols/jsonrpc.py` -> PDA UI.

## 7. Bagaimana Modul Dipanggil Saat Server Jalan

Saat server start:
1. Tryton baca config dari `config/trytond.conf`.
2. Tryton connect ke DB.
3. Modul yang ter-install pada DB di-load ke Pool.
4. Object model/wizard/report dari modul tersedia untuk RPC/UI.

Script yang sering dipakai:
- `scripts/install.sh`: pasang/symlink module source ke environment Tryton.
- `scripts/update.sh`: update modul ke database.
- `scripts/run.sh`: jalankan server Tryton.

## 8. Alur Development Feature Baru (Simple SOP)

Checklist cepat:
1. Tentukan scope: masuk `wms`, `lotte`, atau `file_import`.
2. Buat branch dari `develop`.
3. Implement logic Python (model/wizard/report).
4. Tambah XML view/action/menu/security.
5. Daftarkan object + XML di `tryton.cfg` modul.
6. Jalankan update modul ke DB.
7. Tes manual dari UI + tes otomatis.
8. PR ke `develop`.

Command umum:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/LWF02-xxx-nama-feature

bash /projects/scripts/install.sh

trytond-admin \
  -c /projects/config/trytond.conf \
  --logconf /projects/config/trytond.log.conf \
  -d wms \
  -u lotte

python -m pytest /projects/src/lotte/tests/
```

## 9. Mapping Praktis untuk Developer MVC

Kalau Anda terbiasa MVC, pakai mapping ini:
- Controller -> method pada model/wizard Tryton.
- Route URL -> method RPC (`model.*`, `wizard.*`, `report.*`).
- View template -> XML view Tryton (`form`, `tree`) + metadata action/menu.
- Middleware/transaction -> dikelola framework Tryton di dispatcher.

## 10. Starting Point Saat Dapat Tiket Baru

Urutan baca paling efektif:
1. Baca `tryton.cfg` modul target untuk tahu file yang dipakai.
2. Baca XML (`menu`, `action`, `view`) untuk tahu entry UI.
3. Baca file Python model/wizard terkait untuk logic.
4. Jalankan dari UI dan lihat tombol/state yang memicu method.
5. Tambah test di `tests/` modul.

## 11. Ringkasan Singkat

- Di proyek ini, flow utama bisnis ada di modul Tryton dalam `src/`.
- UI ditentukan XML, logic dieksekusi method model/wizard/report.
- "Routing" utama aplikasi adalah RPC method, bukan URL controller tradisional.
- `lotte` adalah layer custom di atas `wms` dan `file_import`.

Kalau sudah nyaman dengan flow ini, langkah berikutnya adalah membuat 1 feature kecil end-to-end di `lotte` untuk latihan (1 field baru + 1 tombol + 1 test).
