import React from 'react'
import { FileTree } from 'nextra/components'

// スタイル定義
const highlightStyle = { color: "#991b1b", textDecoration: "none", fontWeight: "bold" };
const linkStyleBase = { textDecoration: "underline", cursor: "pointer", display: "inline-flex", alignItems: "baseline", gap: "4px" };
const linkDefaultColor = { color: "#0070f3" };

export const ProjectStructure = ({ highlights = [] }: { highlights?: string[] }) => {

    // ヘルパー関数: urlがあればリンク、なければテキスト(強調判定あり)
    const f = (name: string, url?: string) => {
        const isHighlighted = highlights.includes(name);

        if (url) {
            const finalLinkStyle = {
                ...linkStyleBase,
                ...(isHighlighted ? highlightStyle : linkDefaultColor)
            };
            return (
                <a href={url} target="_blank" rel="noopener noreferrer" style={finalLinkStyle}>
                    {name}
                    <svg fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" height="0.8em" style={{ display: 'inline', verticalAlign: 'baseline' }}>
                        <path d="M7 17L17 7"></path><path d="M7 7h10v10"></path>
                    </svg>
                </a>
            );
        }

        return isHighlighted ? <span style={highlightStyle}>{name}</span> : name;
    };

    return (
        <FileTree>
            <FileTree.Folder name="KptSort" defaultOpen>
                <FileTree.Folder name="datasets">
                    <FileTree.Folder name="datasets">
                        <FileTree.Folder name="images">
                            <FileTree.File name={f("train")} />
                            <FileTree.File name={f("val")} />
                        </FileTree.Folder>
                        <FileTree.Folder name="labels">
                            <FileTree.File name={f("train")} />
                            <FileTree.File name={f("val")} />
                        </FileTree.Folder>
                    </FileTree.Folder>
                </FileTree.Folder>

                <FileTree.Folder name="debug_frames">{""}</FileTree.Folder>

                <FileTree.Folder name="installer" defaultOpen>
                    <FileTree.File name={f("environment.sh", "https://github.com/user/repo/blob/main/environment.sh")} />
                    <FileTree.File name={f("install_cv2_w_ffmpeg.sh", "https://github.com/user/repo/blob/main/install_cv2_w_ffmpeg.sh")} />
                    <FileTree.File name={f("requirements.txt", "https://github.com/user/repo/blob/main/requirements.txt")} />
                </FileTree.Folder>

                <FileTree.Folder name="KptSORT-docs">
                    <FileTree.Folder name="app">
                        <FileTree.File name={f("[[...mdxPath]]")} />
                    </FileTree.Folder>
                    <FileTree.Folder name="content">{""}</FileTree.Folder>
                    <FileTree.Folder name="patches">{""}</FileTree.Folder>
                    <FileTree.Folder name="public">{""}</FileTree.Folder>
                </FileTree.Folder>

                <FileTree.Folder name="output">
                    <FileTree.Folder name="{file_name}">
                        <FileTree.File name={f("{threshould}_{filename}_{frames}.mp4")} />
                        <FileTree.File name={f("bees.pkl", "/KptSORT/Start/")} />
                        <FileTree.File name={f("data_graph.pkl")} />
                        <FileTree.File name={f("exchanged_map.png")} />
                        <FileTree.File name={f("exchanged_series.png")} />
                        <FileTree.File name={f("gt.txt")} />
                        <FileTree.File name={f("hive_heatmap.png")} />
                        <FileTree.File name={f("hived_counter.png")} />
                        <FileTree.File name={f("hived_series.png")} />
                        <FileTree.File name={f("img_tracklets.png")} />
                        <FileTree.File name={f("trackers.npz")} />
                        <FileTree.File name={f("trackrets.png")} />
                        <FileTree.File name={f("trajectories.png")} />
                        <FileTree.File name={f("trajectories_med_series.pkl")} />
                        <FileTree.File name={f("trajectories_med_series.png")} />
                        <FileTree.File name={f("trajectories_series.pkl")} />
                        <FileTree.File name={f("trajectories_series.png")} />
                    </FileTree.Folder>
                </FileTree.Folder>

                <FileTree.Folder name="result">
                    <FileTree.File name={f("pps64_cnl3_1")} />
                </FileTree.Folder>

                <FileTree.Folder name="runs">
                    <FileTree.Folder name="obb">
                        <FileTree.Folder name="train">
                            <FileTree.File name={f("weights")} />
                        </FileTree.Folder>
                    </FileTree.Folder>
                </FileTree.Folder>

                <FileTree.Folder name="sources">
                    <FileTree.Folder name="{file_name}">
                        <FileTree.File name={f("{file_name}.mp4")} />
                        <FileTree.File name={f("BU.pickle")} />
                        <FileTree.File name={f("CTD.csv")} />
                    </FileTree.Folder>
                    <FileTree.Folder name="hives">
                        <FileTree.Folder name="{hive_name}">
                            <FileTree.File name={f("{hive_name}.pickle")} />
                            <FileTree.File name={f("{hive_name}.png")} />
                            <FileTree.File name={f("result_{hive_name}.png")} />
                        </FileTree.Folder>
                    </FileTree.Folder>
                    <FileTree.Folder name="Models">
                        <FileTree.File name={f("best.pt")} />
                        <FileTree.File name={f("sam_vit_h_4b8939.pth")} />
                    </FileTree.Folder>
                </FileTree.Folder>

                <FileTree.Folder name="tools">
                    <FileTree.File name={f("AssignBeeHive.py", "https://github.com/...")} />
                    <FileTree.File name={f("calk_oks.py")} />
                    <FileTree.File name={f("figs.pkl")} />
                    <FileTree.File name={f("generate_graph.py")} />
                    <FileTree.File name={f("generate_hive.py")} />
                    <FileTree.File name={f("kpsort.py", "/docs/tools/kpsort")} />
                    <FileTree.File name={f("loadpkl_jit.py")} />
                </FileTree.Folder>

                <FileTree.File name={f(".gitignore")} />
                <FileTree.File name={f(".gitlab-ci.yml")} />
                <FileTree.File name={f("analysis.py")} />
                <FileTree.File name={f("dashboard.py")} />
                <FileTree.File name={f("tracking.py")} />
                <FileTree.File name={f("train.py")} />
                <FileTree.File name={f("train2.yaml")} />
            </FileTree.Folder>
        </FileTree>
    );
};