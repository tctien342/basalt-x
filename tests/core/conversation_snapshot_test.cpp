#include <algorithm>
#include "basalt/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <future>
#include <vector>

using namespace basalt::core;
using namespace basalt::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}

void full_session(int fmt, int mode, int experts) {
    Fixture main(fmt,mode), draft(fmt==3?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();   // the whole-model carve
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"whole-session geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        // A real indexer maintains its spare pooled row as a copy of idx_dead.
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(),ids.begin()+3);
    check(conversation_checkpoint_save(checkpoint,ss,g,err),"save complete running checkpoint");
    {   // a deferred save equals the plain one, though the device state changes right after it is issued
        ConversationCheckpointDeferral deferral;
        check(deferral.reserve(sizes.gdn,err),"reserve the deferred checkpoint buffer");
        ConversationCheckpoint deferred;
        deferred.ids=checkpoint.ids;
        check(deferral.save(deferred,ss,g,err) && deferral.pending()==deferred.state_a.data(),"issue deferred checkpoint save");
        cuda_check(cudaMemset(ss.gdn_state,0x5a,sizes.gdn));
        cuda_check(cudaDeviceSynchronize());
        check(deferral.finish(err) && deferral.pending()==nullptr,"finish deferred checkpoint save");
        check(deferred.state_a.size()==checkpoint.state_a.size() &&
              std::equal(deferred.state_a.begin(),deferred.state_a.end(),checkpoint.state_a.begin()) &&
              std::equal(deferred.state_b.begin(),deferred.state_b.end(),checkpoint.state_b.begin()) &&
              std::equal(deferred.state_c.begin(),deferred.state_c.end(),checkpoint.state_c.begin()) &&
              std::equal(deferred.state_d.begin(),deferred.state_d.end(),checkpoint.state_d.begin()) &&
              std::equal(deferred.state_e.begin(),deferred.state_e.end(),checkpoint.state_e.begin()),
              "deferred checkpoint bytes match the plain save");
        cuda_check(cudaMemset(ss.gdn_state,13,sizes.gdn));
        cuda_check(cudaDeviceSynchronize());
    }
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(conversation_snapshot_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    if (fmt == kKvInt8 && mode == 1) {
        std::vector<uint32_t> directory;
        for (uint32_t page = 0; page < 17; ++page) directory.push_back((page * 5) % 24);
        main.state.shared_residency = true;
        main.state.host.fp16_host_pages = main.state.map.page_table;
        main.state.host_page_ids = directory.data();
        main.state.host_page_count = directory.size();
        main.state.host_pool_pages = main.state.n_pages;
        SavedConversation expected;
        check(conversation_snapshot_save(expected, view, ss, g, draft.state, err), "synchronous shared snapshot");
        auto chain = [&] {
            SavedConversation value;
            value.live.ids = ids;
            value.live.imgs = images;
            value.checkpoints.push_back(checkpoints[0].shared_copy());
            return value;
        };
        ConversationSharedSnapshot sealed;
        check(conversation_snapshot_seal_shared(sealed, chain(), ss, g, &draft.state, err),
              "seal shared INT8 state and draft");
        check(sealed.image.kv[0].k.empty() && !sealed.image.kv[1].k.empty(),
              "sealing leaves target K/V for the worker");
        ConversationSharedSnapshot frozen;
        check(conversation_snapshot_seal_shared(frozen, chain(), ss, g, &draft.state, err, true) &&
              frozen.image.kv[0].pooled.empty() && !frozen.image.kv[1].k.empty(),
              "leave the leased pooled rows for deferred read");
        ConversationSharedSnapshot split_draft;
        check(conversation_snapshot_seal_shared(split_draft, chain(), ss, g, &draft.state, err, true, 4) &&
              split_draft.draft_tail.first_cell == 4 && split_draft.draft_tail.end_cell > 4,
              "seal page-rounded draft prefix and retain the high tail");
        SavedConversation frozen_image;
        check(!conversation_snapshot_finish_shared(frozen_image, std::move(frozen), err),
              "refuse to publish before reading the pooled rows");
        directory[0] = 23;
        float* other_index = nullptr;
        main.alloc(other_index, main.sizes[4]);
        cuda_check(cudaMemset(other_index, 0x77, main.sizes[4]));
        auto index_worker = std::async(std::launch::async, [&] {
            return conversation_snapshot_read_shared_index(frozen, err) &&
                   conversation_snapshot_finish_shared(frozen_image, std::move(frozen), err);
        });
        cuda_check(cudaMemset(other_index, 0x33, main.sizes[4]));
        check(index_worker.get() && equal(frozen_image.kv[0], expected.kv[0]) &&
              equal(frozen_image.kv[1], expected.kv[1]),
              "frozen index reads the old slot, including the spare row, through its own stream");
        for (size_t i = 0; i < 4; ++i)
            if (draft.sizes[i])
                cuda_check(cudaMemset(draft.sources[i], 0x27, draft.sizes[i] / draft.state.max_cells * 4));
        cuda_check(cudaDeviceSynchronize());
        SavedConversation split_image;
        auto split_worker = std::async(std::launch::async, [&] {
            return conversation_snapshot_read_shared_draft(split_draft, err) &&
                   conversation_snapshot_read_shared_index(split_draft, err) &&
                   conversation_snapshot_finish_shared(split_image, std::move(split_draft), err);
        });
        check(split_worker.get() && equal(split_image.kv[1], expected.kv[1]),
              "draft prefix rewrites preserve exact sealed bytes and deferred tail");
        cuda_check(cudaMemset(main.state.idx_pooled, 0x3F, main.sizes[4]));
        draft.fill(81);
        SavedConversation deferred;
        auto worker = std::async(std::launch::async, [&] {
            return conversation_snapshot_finish_shared(deferred, std::move(sealed), err);
        });
        check(worker.get() && deferred.live.ids == expected.live.ids && deferred.live.state_a == expected.live.state_a &&
              deferred.live.state_e == expected.live.state_e && deferred.checkpoints[0].used == 17 &&
              equal(deferred.kv[0], expected.kv[0]) && equal(deferred.kv[1], expected.kv[1]),
              "worker uses the frozen directory and sealed device state and draft");
        main.state.shared_residency = false;
        main.state.host.fp16_host_pages = nullptr;
        main.state.host_page_ids = nullptr;
        main.state.host_page_count = main.state.host_pool_pages = 0;
    }
    fill(177);
    check(conversation_snapshot_save(b,view,ss,g,draft.state,err),"capture complete B");
    check(a.live.state_d!=b.live.state_d,"different opening-state spare keys in regression fixture");
    auto bad=a; bad.kv.back().k.pop_back();
    check(conversation_snapshot_restore(bad,ss,g,draft.state,err)==ConversationRestore::invalid,
          "reject invalid late draft before any main-layer write");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.state_a==b.live.state_a && restored.live.state_d==b.live.state_d &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(restored.live.state_a==a.live.state_a && restored.live.state_b==a.live.state_b && restored.live.state_c==a.live.state_c &&
          restored.live.state_d==a.live.state_d && restored.live.state_e==a.live.state_e &&
          equal(restored.kv[0],a.kv[0]) && equal(restored.kv[1],a.kv[1]),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    for (int64_t dirty : {65, 3, 0}) {
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore growth fixture base");
        ConversationKvReuse reuse{a.kv,65,dirty,{}};
        const uint8_t* original = nullptr;
        reuse.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){original=p;return true;});
        main.fill_after(91,dirty); draft.fill_after(91,std::max<int64_t>(0,dirty-1));
        cuda_check(cudaMemset(ss.gdn_state,91,sizes.gdn));
        ids.resize(70);
        for (size_t i=65;i<ids.size();++i) ids[i]=int32_t(i+1);
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        SavedConversation fresh,incremental;
        check(conversation_snapshot_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        check(conversation_snapshot_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
        check(incremental.bytes() <= peak,"incremental allocation stays within admitted bound");
        check(incremental.live.state_a==fresh.live.state_a && incremental.live.state_b==fresh.live.state_b &&
              incremental.live.state_d==fresh.live.state_d && equal(incremental.kv[0],fresh.kv[0]) &&
              equal(incremental.kv[1],fresh.kv[1]),"incremental capture equals full capture after growth or rewind");
        check((dirty>=4)==(reused>0),"only complete unchanged pages or rows are retained");
        incremental.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){
            check(p==original,"growth never reallocates the retained payload");return true;
        });
        check(conversation_snapshot_restore(incremental,ss,g,draft.state,err)==ConversationRestore::restored,"restore segmented incremental snapshot");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(incremental.kv.back(),draft.state,g,70,false,fingerprint,err),"incremental draft authoritative and ring read-back");
        ids.resize(65);
    }
    {
        // a layer split's later stage: the same image WITHOUT the draft layer's K/V (draft == nullptr)
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A before stage images");
        size_t with_draft=0,without=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,with_draft,err) &&
              conversation_snapshot_bytes(view,ss,g,nullptr,without,err) && without<with_draft,
              "a stage image's estimate leaves the draft out");
        SavedConversation stage,stage_b,stage_back;
        check(conversation_snapshot_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.state_a==a.live.state_a,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(conversation_snapshot_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.state_a==stage_b.live.state_a &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.state_a==stage.live.state_a &&
              stage_back.live.state_d==stage.live.state_d && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(conversation_snapshot_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        check(conversation_snapshot_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
        check(reused>0 && incremental.bytes()<=peak && equal(incremental.kv[0],fresh.kv[0]) &&
              incremental.live.state_a==fresh.live.state_a,"an incremental stage capture reuses pages and equals a full one");
        SavedConversation split_image=stage;
        split_image.stage_images.push_back(stage);
        check(!conversation_snapshot_validate(split_image,ss,g,draft.state,err),
              "a layer split's image is refused by the whole-session form");
        ConversationKvReuse wrong{a.kv,65,65,{}};
        check(!conversation_snapshot_capture_bytes(wrong,view,ss,g,nullptr,peak,err),"a draft image's K/V is not a stage's reuse");
    }
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A after stage images");
    check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"restore early running checkpoint");
    std::vector<uint8_t> spare(sizes.dead);
    cuda_check(cudaMemcpy(spare.data(),main.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
    check(std::equal(spare.begin(),spare.end(),a.checkpoints[0].state_d.begin(),a.checkpoints[0].state_d.end()),"checkpoint rebuilds spare row over a later completed block");
    check(ss.ple_prev[0]==2 && ss.ple_prev[1]==3,"checkpoint PLE token window");
}

// The same logical image can move between different physical host page orders.
void mapped_kv(int format, bool rotation) {
    Fixture source(format, 1), target(format, 1);
    source.state.kv_rot = target.state.kv_rot = rotation;
    uint32_t source_pages[] = {8, 2, 3}, target_pages[] = {5, 6, 1};
    auto bind = [](Fixture& f, const uint32_t (&pages)[3]) {
        int32_t* device = nullptr;
        f.alloc(device, sizeof(pages));
        cuda_check(cudaMemcpy(device, pages, sizeof(pages), cudaMemcpyHostToDevice));
        f.state.host.fp16_host_pages = device;
        f.state.host_page_ids = pages;
        f.state.host_page_count = 3;
        f.state.host_pool_pages = f.state.n_pages;
    };
    bind(source, source_pages);
    bind(target, target_pages);
    source.fill(42);
    target.fill(73);
    std::string error;
    ConversationKv image, copied;
    constexpr int64_t tokens = 9;
    check(conversation_kv_save(image, source.state, source.g, tokens, true, error), "capture mapped K/V pages");
    const std::array<const ConversationBuffer*, 4> payloads = {&image.k, &image.v, &image.k_scale, &image.v_scale};
    for (size_t part = 0; part < payloads.size(); ++part) {
        const auto& payload = *payloads[part];
        if (payload.empty()) continue;
        const size_t page_bytes = payload.size() / 3;
        for (size_t i = 0; i < 3; ++i) {
            std::vector<uint8_t> actual(page_bytes), expected(page_bytes);
            check(payload.read(actual.data(), i * page_bytes, page_bytes), "read logical saved page");
            const auto* bytes = static_cast<const uint8_t*>(source.sources[part]);
            cuda_check(cudaMemcpy(expected.data(), bytes + source_pages[i] * page_bytes, page_bytes, cudaMemcpyDefault));
            check(actual == expected, "logical snapshot uses its physical page for each payload");
        }
    }
    uint64_t fingerprint = 0;
    check(conversation_kv_restore(image, target.state, target.g, tokens, true, error), "restore to another mapping");
    check(conversation_kv_verify(image, target.state, target.g, tokens, true, fingerprint, error), "verify mapped restore");
    target.fill(101);
    check(conversation_kv_copy(source.state, target.state, target.g, tokens, true, error), "copy between distinct mappings");
    check(conversation_kv_save(copied, target.state, target.g, tokens, true, error) && equal(image, copied),
          "peer copy preserves the logical payload");
    target_pages[2] = target.state.n_pages;
    check(!conversation_kv_restore(image, target.state, target.g, tokens, true, error), "reject page outside pool");
}

void selective_staging(int format) {
    Fixture source(format, 1), stage(format, 0);
    source.fill(42);
    stage.fill(73);
    const auto& st = source.state;
    const std::array<void*, 4> gpu = format == kKvInt8
        ? std::array<void*, 4>{st.k_q, st.v_q, st.k_scale, st.v_scale}
        : std::array<void*, 4>{st.k_pool, st.v_pool, nullptr, nullptr};
    int32_t* mapping = nullptr;
    int32_t* ids = nullptr;
    int32_t* steps = nullptr;
    uint32_t* bitmap = nullptr;
    const int32_t map[] = {8, 2, 3};
    const int32_t cells[] = {0, 8, 12};  // two selected cells are historical
    std::array<int32_t, kStepCount> row{};
    row[kStepWidth] = 3;
    source.alloc(mapping, sizeof(map));
    source.alloc(ids, sizeof(cells));
    source.alloc(steps, sizeof(row));
    source.alloc(bitmap, sizeof(uint32_t));
    cuda_check(cudaMemcpy(mapping, map, sizeof(map), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(ids, cells, sizeof(cells), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(steps, row.data(), sizeof(row), cudaMemcpyHostToDevice));
    source.state.host.fp16_host_pages = mapping;
    std::vector<int32_t> resident(st.n_pages, -1);
    resident[3] = 1;
    cuda_check(cudaMemcpy(st.page_table, resident.data(), resident.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    for (size_t part = 0; part < gpu.size(); ++part)
        if (gpu[part]) {
            const size_t bytes = source.sizes[part] / st.n_pages;
            cuda_check(cudaMemset(static_cast<uint8_t*>(gpu[part]) + bytes, 90 + part, bytes));
        }
    QsaAttnPools src;
    src.k_pool = st.k_pool; src.v_pool = st.v_pool;
    src.k_q = st.k_q; src.v_q = st.v_q; src.k_scale = st.k_scale; src.v_scale = st.v_scale;
    src.page_table = st.page_table; src.physical_pages = mapping;
    QsaAttnPools dst;
    dst.k_pool = stage.state.k_pool; dst.v_pool = stage.state.v_pool;
    dst.k_q = stage.state.k_q; dst.v_q = stage.state.v_q;
    dst.k_scale = stage.state.k_scale; dst.v_scale = stage.state.v_scale;
    check(kv_stage_selected(dst, src, st.host, format, qsa_real_shapes(), bitmap, ids, steps, 1, 3, 9, nullptr),
          "launch selective staging");
    for (size_t part = 0; part < gpu.size(); ++part) {
        if (!gpu[part]) continue;
        const size_t bytes = source.sizes[part] / st.n_pages;
        std::vector<uint8_t> actual(bytes), expected(bytes);
        const auto* staged = static_cast<const uint8_t*>(stage.sources[part]);
        const auto* host = static_cast<const uint8_t*>(source.sources[part]);
        cuda_check(cudaMemcpy(actual.data(), staged, bytes, cudaMemcpyDefault));
        cuda_check(cudaMemcpy(expected.data(), host + 8 * bytes, bytes, cudaMemcpyDefault));
        check(actual == expected, "cache miss stages the mapped host page");
        cuda_check(cudaMemcpy(actual.data(), staged + 2 * bytes, bytes, cudaMemcpyDefault));
        check(std::all_of(actual.begin(), actual.end(), [&](uint8_t x) { return x == 90 + part; }),
              "cache hit stages the resident physical slot");
        std::vector<uint8_t> original(stage.sizes[part]);
        for (size_t j = 0; j < original.size(); ++j) original[j] = uint8_t(73 + part * 31 + j * 7 + j / 257);
        for (int page : {1, 3}) {
            cuda_check(cudaMemcpy(actual.data(), staged + page * bytes, bytes, cudaMemcpyDefault));
            check(std::equal(actual.begin(), actual.end(), original.begin() + page * bytes),
                  "unselected or non-historical page remains unchanged");
        }
    }
}

// An elastic state backs a prefix of its cells: a snapshot longer than that is refused, not copied past it.
void elastic_backing() {
    KvArena arena;
    std::string err;
    constexpr int64_t kMiB = 1 << 20;
    if (!arena.open({(uint64_t) kMiB}, 24, 4, err)) {
        std::printf("elastic backing: skipped (%s)\n", err.c_str());
        return;
    }
    check(arena.map(4, err) && arena.cells() >= 4 && arena.cells() < 65,"back a short prefix");
    Fixture f(kKvInt8,0);
    f.fill(13);
    cuda_check(cudaDeviceSynchronize());
    ConversationKv image;
    check(conversation_kv_save(image,f.state,f.g,65,true,err),"snapshot a long conversation");
    f.state.kv_arena=&arena;
    check(!conversation_kv_restore(image,f.state,f.g,65,true,err) && err.find("backed")!=std::string::npos,
          "restore past the backed cells is refused");
    ConversationKv again;
    check(!conversation_kv_save(again,f.state,f.g,65,true,err),"save past the backed cells is refused");
    check(arena.map(65,err) && conversation_kv_restore(image,f.state,f.g,65,true,err),"restore once the cells are backed");
    check(conversation_kv_restore(image,f.state,f.g,3,true,err)==false,"a shorter extent than the image is incompatible");
    f.state.kv_arena=nullptr;
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    elastic_backing();
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for (int mode : {0,1,2}) {
        if (fmt==3 && mode!=0) continue;
        Fixture f(fmt,mode);
        if (fmt==3) {
            std::string err;
            ConversationKv invalid;
            for (int unsupported : {1,2}) {
                f.state.kv_mode=unsupported;
                check(conversation_kv_bytes(f.state,f.g,9,true)==0,"hybrid streaming/ring rejected without block-mover abort");
                check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),"hybrid streaming/ring capture fails closed");
            }
            f.state.kv_mode=0;
        }
        {
            std::string err;
            ConversationKv invalid;
            for (const int64_t bad : {-1LL, 97LL, (long long) std::numeric_limits<int64_t>::max()}) {
                check(!conversation_kv_save(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid save extent before allocation");
                check(!conversation_kv_restore(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid restore extent before copying");
                check(conversation_kv_bytes(f.state,f.g,bad,mode!=2)==0,"invalid extent cannot overflow byte estimate");
            }
        }
        for (int64_t upto : {0,1,3,4,5,63,64,65,96}) {
            const bool index=mode!=2;
            std::string err;
            f.fill(13);
            ConversationKv a,b,restored;
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(a,f.state,f.g,upto,index,err),"save A");
            check(a.bytes()==conversation_kv_bytes(f.state,f.g,upto,index),"size estimate equals snapshot payload");
            f.fill(177);
            check(conversation_kv_save(b,f.state,f.g,upto,index,err),"save B");
            check(upto==0 || !equal(a,b),"fixture changes state");
            check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"restore A over B");
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(restored,f.state,f.g,upto,index,err),"read back restored A");
            check(equal(a,restored),"A/B/A byte-exact K/V and indexer state");
            uint64_t fingerprint = 0;
            check(conversation_kv_verify(a,f.state,f.g,upto,index,fingerprint,err),"verify restored authoritative and resident bytes without rewriting them");
            {
                Fixture h(fmt,mode);   // another session of the same layout: A copied into it on the device
                h.fill(177);
                cuda_check(cudaDeviceSynchronize());
                check(conversation_kv_copy(f.state,h.state,f.g,upto,index,err),"copy A into another session");
                cuda_check(cudaDeviceSynchronize());
                ConversationKv copied;
                check(conversation_kv_save(copied,h.state,h.g,upto,index,err),"read back the copy");
                check(equal(a,copied),"device copy byte-exact K/V and indexer state");
                uint64_t copied_fp = 0;
                check(conversation_kv_verify(a,h.state,h.g,upto,index,copied_fp,err),"verify the copy's authoritative and resident bytes");
            }
            if (mode==1) {
                std::vector<int32_t> table((size_t)f.state.n_pages);
                cuda_check(cudaMemcpy(table.data(),f.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
                check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"stale VRAM pages invalidated");
            }
            if (mode==2 && upto>0) {
                const auto s=qsa_real_shapes();
                const int64_t b1=(upto+s.page_size-1)/s.page_size,b0=std::max<int64_t>(0,b1-f.state.n_slots);
                const size_t block=kv_block_bytes(s,fmt)/2;
                // Codes and scales are separate in INT8; check code bytes here.
                const size_t code_block=fmt==kKvInt8 ? s.page_size*s.n_head_kv*s.head_dim : block;
                const void* slots=fmt==kKvQ4?(void*)f.state.k_q4:fmt==kKvInt8?(void*)f.state.k_q:(void*)f.state.k_pool;
                std::vector<uint8_t> got(code_block);
                for (int64_t page=b0;page<b1;++page) {
                    cuda_check(cudaMemcpy(got.data(),(const uint8_t*)slots+(page%f.state.n_slots)*code_block,code_block,cudaMemcpyDeviceToHost));
                    std::vector<uint8_t> expected(code_block);
                    check(a.k.read(expected.data(), page*code_block, code_block),"read segmented expected draft page");
                    check(got == expected,"draft ring contains restored pages");
                }
            }
            auto bad=a; bad.head_dim++;
            check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject incompatible geometry");
            if (fmt==kKvInt8) {
                // INT8 K/V saved through the Hadamard rotation (KV_ROT=1) never restores into a state
                // without it, nor the reverse
                f.state.kv_rot=true;
                ConversationKv rotated;
                check(conversation_kv_save(rotated,f.state,f.g,upto,index,err),"save rotated INT8");
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image into a rotated state rejected");
                f.state.kv_rot=false;
                check(!conversation_kv_restore(rotated,f.state,f.g,upto,index,err),"rotated image into an unrotated state rejected");
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image still restores");
            }
            if (!a.k.empty()) {
                bad=a; bad.k.pop_back();
                check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject malformed payload");
            }
        }
    }
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) for (int experts : {256,512})
        full_session(fmt,mode,experts);
    for (int experts : {256,512}) full_session(3,0,experts);
    mapped_kv(kKvF16, false);
    mapped_kv(kKvInt8, true);
    selective_staging(kKvF16);
    selective_staging(kKvInt8);
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
